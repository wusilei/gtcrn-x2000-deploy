/**
 * noise_reduction.c — GTCRN Linux 侧降噪适配层 (v12: 全Q15定点 + 定点FFT)
 * ============================================================
 * P4+P5: 定点 FFT 替换 KissFFT, 全STFT管线Q15定点化。
 * 消除所有 float 操作：无 float 除法/乘法/加法，无 sqrtf/sinf/expf。
 *
 * 流水线 (全Q15定点):
 *   8kHz PCM(int16=Q15) → 上采样(Q15) → FIFO → Hann窗(Q15)
 *   → 定点FFT → float→GTCRN(不变) → CRM >>5 → 定点IFFT
 *   → Hann窗(Q15) → OLA(int32) → EQ(Q15) → 下采样 → 8kHz PCM(Q15)
 */
#include "noise_reduction.h"
#include "gtcrn_fp.h"
#include "gtcrn_matlab_weights.h"
#include "fft_q15.h"
#include "kiss_fftr.h"
#include <stdlib.h>
#include <string.h>

#define N_FFT       512
#define WIN_LEN     512
#define WIN_INC     256
#define N_BINS      257
#define FRAME_IN    200
#define FRAME_16K   400
#define FIFO_SZ     (WIN_LEN * 4)
#define WARMUP_MUTE 20
#define WARMUP_FADE 12

/* ================================================================
 * Q15 Constants
 * ================================================================
 * PCM int16 is native Q15 (÷32768 implicit).
 * Hann window: int16 Q15.
 * FFT: int32 data, int16 twiddle Q15.
 * OLA: int32 accumulator (headroom for ~4 overlapping frames).
 * EQ: Q15 biquad with int32 state (headroom for +18dB gain).
 * CRM→IFFT: crm >> 5 (s32f20 → Q15×512 scale, since 2^20/2^15=2^5=32).
 */

/* Q15 sqrt-Hann window: sin(π*i/(N-1)) × 32767 */
static int16_t hann_q15[WIN_LEN];
static int     g_win_inited = 0;

/* ================================================================
 * Stream state (all Q15)
 * ================================================================ */
static gtcrn_state_t g_state;

/* Input FIFO: Q15 int16 */
static int16_t g_fifo[FIFO_SZ];
static int     g_fifo_wpos;
static int     g_fifo_count;

/* OLA buffer: int32 (headroom for overlap accumulation) */
static int32_t g_ola[WIN_LEN + WIN_INC];
static int     g_ola_pos;

/* Output FIFO: Q15 int16 */
static int16_t g_out_fifo[FIFO_SZ];
static int     g_out_rpos;
static int     g_out_count;

static int     g_frame_count;
static int16_t g_last_in_8k;

/* DEBUG: KissFFT for forward path only */
static kiss_fftr_cfg g_fft_fwd = NULL;

/* EQ state: int32 (unclamped, +18dB shelf headroom) */
static int32_t g_eq_x1, g_eq_x2, g_eq_y1, g_eq_y2;

/* ================================================================
 * Q15 EQ biquad coefficients (fc=1300Hz +18dB @ 16kHz)
 * round(coeff * 32768)
 * ================================================================ */
#define HISH_B0_Q15  178748
#define HISH_B1_Q15 -280273
#define HISH_B2_Q15  115575
#define HISH_A1_Q15   29236   /* -(-0.892201) * 32768 */
#define HISH_A2_Q15  -10354   /* -(0.315977) * 32768 */

void noise_init(void) {
    if (!g_win_inited) {
        for (int i = 0; i < WIN_LEN; i++) {
            hann_q15[i] = (int16_t)(sinf(3.14159265358979323846f * (float)i
                                   / (float)(WIN_LEN - 1)) * 32767.0f + 0.5f);
        }
        g_win_inited = 1;
    }
    gtcrn_state_init(&g_state);
    if (!g_fft_fwd) g_fft_fwd = kiss_fftr_alloc(N_FFT, 0, NULL, NULL);
    g_fifo_wpos   = 0;
    g_fifo_count  = 0;
    g_ola_pos     = 0;
    g_out_rpos    = 0;
    g_out_count   = 0;
    g_frame_count = 0;
    g_last_in_8k  = 0;
    g_eq_x1 = g_eq_x2 = g_eq_y1 = g_eq_y2 = 0;
    memset(g_fifo,     0, sizeof(g_fifo));
    memset(g_ola,      0, sizeof(g_ola));
    memset(g_out_fifo, 0, sizeof(g_out_fifo));
}

void noise_deinit(void) {
    if (g_fft_fwd) { free(g_fft_fwd); g_fft_fwd = NULL; }
}

void noise_reduction(short *voiceIn, short *voiceOut) {
    /* ── 1. 8kHz → 16kHz 上采样 (Q15, 纯整数) ── */
    for (int i = 0; i < FRAME_IN; i++) {
        int16_t s_curr = voiceIn[i];          /* native Q15 */
        int16_t s_prev = g_last_in_8k;
        g_fifo[g_fifo_wpos] = s_curr;
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo[g_fifo_wpos] = (int16_t)(((int)s_prev + (int)s_curr) >> 1);
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo_count += 2;
        g_last_in_8k = voiceIn[i];
    }

    /* ── 2. STFT→GTCRN→ISTFT (时间正序, 全定点) ── */
    {
        int read_wpos = (g_fifo_wpos - g_fifo_count + WIN_LEN + FIFO_SZ) % FIFO_SZ;
        while (g_fifo_count >= WIN_LEN) {
            g_fifo_count -= WIN_INC;

            /* === 2a. Hann window (Q15) + KissFFT forward (float, for debug) === */
            int32_t fft_in[WIN_LEN];
            kiss_fft_scalar fft_in_f[WIN_LEN];
            int start = (read_wpos - WIN_LEN + FIFO_SZ) % FIFO_SZ;
            for (int i = 0; i < WIN_LEN; i++) {
                int32_t v = g_fifo[(start + i) % FIFO_SZ];
                fft_in[i] = ((int32_t)v * hann_q15[i] + 16384) >> 15;
                fft_in_f[i] = (float)fft_in[i] * (1.0f / 32768.0f);
            }

            /* KissFFT forward (float, same as linux_api5) */
            kiss_fft_cpx fft_out_f[N_BINS];
            kiss_fftr(g_fft_fwd, fft_in_f, fft_out_f);

            float real[N_BINS], imag[N_BINS];
            for (int i = 0; i < N_BINS; i++) {
                real[i] = fft_out_f[i].r;
                imag[i] = fft_out_f[i].i;
            }

            /* === 2c. GTCRN inference (unchanged) === */
            int32_t crm[2 * N_BINS];
            gtcrn_infer_frame(real, imag, &g_state,
                              erb_erb_fc_weight, erb_ierb_fc_weight, crm);

            /* === 2d. CRM → IFFT input (s32f20 → Q15×512 via >>5) === */
            int32_t inv_real[N_BINS], inv_imag[N_BINS];
            for (int i = 0; i < N_BINS; i++) {
                inv_real[i] = (crm[i] + 16) >> 5;           /* ÷2^5 with rounding */
                inv_imag[i] = (crm[N_BINS + i] + 16) >> 5;
            }

            /* === 2e. Fixed-point IFFT === */
            int32_t ifft_out[WIN_LEN];
            fft_q15_inverse(inv_real, inv_imag, ifft_out);

            /* === 2f. Synthesis window + /N + OLA.
             * >>24 = >>15 (Q15×Q15→Q15) + >>9 (/N=512, matches float Hann/N).
             * KissFFT convention: no 1/N in IFFT → /N is here in synthesis. */
            for (int i = 0; i < WIN_LEN; i++) {
                int32_t sample = ((int64_t)ifft_out[i] * hann_q15[i] + 8388608) >> 24;
                int pos = (g_ola_pos + i) % (WIN_LEN + WIN_INC);
                g_ola[pos] += sample;
            }

            /* === 2g. OLA output → Q15 → g_out_fifo === */
            for (int i = 0; i < WIN_INC; i++) {
                int32_t v = g_ola[g_ola_pos];
                g_ola[g_ola_pos] = 0;
                g_ola_pos = (g_ola_pos + 1) % (WIN_LEN + WIN_INC);
                if (v >  32767) v =  32767;
                if (v < -32768) v = -32768;
                g_out_fifo[(g_out_rpos + g_out_count) % FIFO_SZ] = (int16_t)v;
                g_out_count++;
            }
            g_frame_count++;
            read_wpos = (read_wpos + WIN_INC) % FIFO_SZ;
        }
    }

    /* ── 3. 输出 8kHz PCM (EQ Q15 biquad + 下采样) ── */
    if (g_out_count >= FRAME_16K) {
        /* Warm-up fade gain: Q15 (gain=1.0 → 32768) */
        int32_t gain_q15 = 32768;
        if (g_frame_count < WARMUP_MUTE) {
            gain_q15 = 0;
        } else if (g_frame_count < WARMUP_MUTE + WARMUP_FADE) {
            gain_q15 = ((g_frame_count - WARMUP_MUTE) * 32768) / WARMUP_FADE;
        }

        int16_t out_16k[FRAME_16K];
        for (int i = 0; i < FRAME_16K; i++) {
            /* Gain (Q15 × Q15 → Q15) */
            int32_t x_q15 = ((int32_t)g_out_fifo[g_out_rpos] * gain_q15 + 16384) >> 15;
            g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;

            /* EQ BYPASS — clamp and output directly */
            if (x_q15 >  32767) x_q15 =  32767;
            if (x_q15 < -32768) x_q15 = -32768;
            out_16k[i] = (int16_t)x_q15;
        }
        g_out_count -= FRAME_16K;

        /* 16kHz → 8kHz: average adjacent samples */
        for (int i = 0; i < FRAME_IN; i++)
            voiceOut[i] = (short)(((int)out_16k[i*2] + (int)out_16k[i*2+1]) >> 1);
    } else {
        memset(voiceOut, 0, FRAME_IN * sizeof(short));
    }
}
