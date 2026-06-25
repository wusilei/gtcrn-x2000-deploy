/**
 * noise_reduction_q15.c — 全Q15 FFT (正向+逆向) + 定点OLA
 * ==========================================================
 * 改动 vs noise_reduction.c (混合版):
 *   - KissFFT 正向 → fft_q15_forward (Q15→float 输出给 GTCRN)
 *   - KissFFT 逆向 → fft_q15_inverse (已是 Q15, 不变)
 *   - OLA: float → int32 (消除 float 累加)
 *
 * 编译: 直接替换 noise_reduction.c
 */
#include "noise_reduction.h"
#include "gtcrn_fp.h"
#include "gtcrn_matlab_weights.h"
#include "fft_q15.h"
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

/* Q15 Hann: sin(π*i/(N-1)) * 32767 */
static int16_t hann_q15[WIN_LEN];
static int16_t hann_div_n_q15[WIN_LEN]; /* hann[i]/N_FFT * 32767 */
static int     g_win_inited = 0;

static gtcrn_state_t g_state;
/* FIFO: Q15 int16 (raw PCM is native Q15) */
static int16_t g_fifo[FIFO_SZ];
static int     g_fifo_wpos, g_fifo_count;
/* OLA: int32 for overlap accumulation */
static int32_t g_ola[WIN_LEN + WIN_INC];
static int     g_ola_pos;
/* Output FIFO: Q15 int16 */
static int16_t g_out_fifo[FIFO_SZ];
static int     g_out_rpos, g_out_count;
static int     g_frame_count;
static int16_t g_last_in_8k;

#define Q15_SCALE  (1.0f / 32768.0f)

void noise_init(void) {
    if (!g_win_inited) {
        for (int i = 0; i < WIN_LEN; i++) {
            hann_q15[i] = (int16_t)(sinf(3.14159265358979323846f * (float)i
                                   / (float)(WIN_LEN - 1)) * 32767.0f + 0.5f);
            hann_div_n_q15[i] = (int16_t)(sinf(3.14159265358979323846f * (float)i
                                         / (float)(WIN_LEN - 1)) / (float)N_FFT * 32767.0f + 0.5f);
        }
        g_win_inited = 1;
    }
    gtcrn_state_init(&g_state);
    g_fifo_wpos = g_fifo_count = 0; g_ola_pos = 0;
    g_out_rpos = g_out_count = 0; g_frame_count = 0; g_last_in_8k = 0;
    memset(g_fifo, 0, sizeof(g_fifo)); memset(g_ola, 0, sizeof(g_ola));
    memset(g_out_fifo, 0, sizeof(g_out_fifo));
}

void noise_deinit(void) { }

void noise_reduction(short *voiceIn, short *voiceOut) {
    /* ── 1. 8kHz → 16kHz 上采样 (纯整数 Q15) ── */
    for (int i = 0; i < FRAME_IN; i++) {
        int16_t s_curr = voiceIn[i];
        int16_t s_prev = g_last_in_8k;
        g_fifo[g_fifo_wpos] = s_curr;
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo[g_fifo_wpos] = (int16_t)(((int)s_prev + (int)s_curr) >> 1);
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo_count += 2;
        g_last_in_8k = voiceIn[i];
    }

    /* ── 2. STFT→GTCRN→ISTFT ── */
    {
        int read_wpos = (g_fifo_wpos - g_fifo_count + WIN_LEN + FIFO_SZ) % FIFO_SZ;
        while (g_fifo_count >= WIN_LEN) {
            g_fifo_count -= WIN_INC;

            /* 2a. Hann window: Q15×Q15 → Q15 */
            int32_t fft_in[WIN_LEN];
            int start = (read_wpos - WIN_LEN + FIFO_SZ) % FIFO_SZ;
            for (int i = 0; i < WIN_LEN; i++) {
                int32_t v = g_fifo[(start + i) % FIFO_SZ];
                fft_in[i] = ((int64_t)v * hann_q15[i] + 16384) >> 15;
            }

            /* 2b. Q15 Forward FFT: Q15 → Q15 output */
            int32_t fwd_r[N_BINS], fwd_i[N_BINS];
            fft_q15_forward(fft_in, fwd_r, fwd_i);

            /* 2c. Q15 → float for GTCRN (requires float input) */
            float real[N_BINS], imag[N_BINS];
            for (int i = 0; i < N_BINS; i++) {
                real[i] = (float)fwd_r[i] * Q15_SCALE;
                imag[i] = (float)fwd_i[i] * Q15_SCALE;
            }

            /* 2d. GTCRN inference */
            int32_t crm[2 * N_BINS];
            gtcrn_infer_frame(real, imag, &g_state,
                              erb_erb_fc_weight, erb_ierb_fc_weight, crm);

            /* 2e. CRM → Q15 IFFT input: s32f20 → Q15 via >>5 */
            int32_t inv_r[N_BINS], inv_i[N_BINS];
            for (int i = 0; i < N_BINS; i++) {
                inv_r[i] = (crm[i] + 16) >> 5;
                inv_i[i] = (crm[N_BINS + i] + 16) >> 5;
            }

            /* 2f. Q15 Inverse FFT */
            int32_t ifft_out[WIN_LEN];
            fft_q15_inverse(inv_r, inv_i, ifft_out);

            /* 2g. Synthesis: Q15×Q15 >>24 → Q15 (/N=512 = >>9, Q15×Q15→Q15 >>15, total >>24) */
            for (int i = 0; i < WIN_LEN; i++) {
                int32_t s = (int32_t)(((int64_t)ifft_out[i] * hann_q15[i] + 8388608) >> 24);
                int pos = (g_ola_pos + i) % (WIN_LEN + WIN_INC);
                g_ola[pos] += s;
            }

            /* 2h. OLA output → clamp Q15 → output FIFO */
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

    /* ── 3. 输出 8kHz PCM ── */
    if (g_out_count >= FRAME_16K) {
        int32_t gain_q15 = 32768;
        if (g_frame_count < WARMUP_MUTE) gain_q15 = 0;
        else if (g_frame_count < WARMUP_MUTE + WARMUP_FADE)
            gain_q15 = ((g_frame_count - WARMUP_MUTE) * 32768) / WARMUP_FADE;

        for (int i = 0; i < FRAME_IN; i++) {
            int32_t a = ((int32_t)g_out_fifo[g_out_rpos] * gain_q15 + 16384) >> 15;
            g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;
            int32_t b = ((int32_t)g_out_fifo[g_out_rpos] * gain_q15 + 16384) >> 15;
            g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;
            int32_t out = (a + b) >> 1; /* downmix 16k→8k */
            if (out >  32767) out =  32767;
            if (out < -32768) out = -32768;
            voiceOut[i] = (int16_t)out;
        }
        g_out_count -= FRAME_16K;
    } else {
        memset(voiceOut, 0, FRAME_IN * sizeof(short));
    }
}
