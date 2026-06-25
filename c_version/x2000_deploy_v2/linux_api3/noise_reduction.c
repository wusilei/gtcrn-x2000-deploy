/**
 * noise_reduction.c — GTCRN Linux 侧降噪适配层 (流式, 参考 libgtcrn_stream)
 * ============================================================
 * v10b: sqrt-Hann 窗 (自动 COLA) + 时序修复
 *
 * 修复:
 *   [1] sqrt-Hann 分析/合成 (Hann COLA, 降噪效果好)
 *   [2] STFT 时间正序 (最旧→最新)
 *   [3] ≥WIN_LEN(512) 才做 STFT
 *   [4] read_wpos 逐帧推进
 *   [5] F2Q20 饱和
 *
 * 8kHz PCM → 2×上采样 → 16kHz FIFO → STFT(sqrt-Hann) → GTCRN
 *   → IFFT ÷N_FFT ×sqrt-Hann → OLA(COLA) → 下采样 → 8kHz PCM
 */
#include "noise_reduction.h"
#include "gtcrn_fp.h"
#include "gtcrn_matlab_weights.h"
#include "kiss_fftr.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_FFT       512
#define WIN_LEN     512
#define WIN_INC     256
#define N_BINS      257
#define FRAME_IN    200
#define FRAME_16K   400
#define FIFO_SZ     (WIN_LEN * 4)
#define WARMUP_MUTE 20
#define WARMUP_FADE 12

/* sqrt-Hann window: sin(π*i/(N-1)) */
static float sqrt_hann_win[WIN_LEN];
static int   g_win_inited = 0;

/* ═══════════════════════════════════════════════════════════
 *  全部流式状态 (static)
 * ═══════════════════════════════════════════════════════════ */
static gtcrn_state_t g_state;
static kiss_fftr_cfg  g_fft_fwd, g_fft_inv;

static float g_fifo[FIFO_SZ];
static int   g_fifo_wpos;
static int   g_fifo_count;

static float g_ola[WIN_LEN + WIN_INC];
static int   g_ola_pos;

static float g_out_fifo[FIFO_SZ];
static int   g_out_rpos;
static int   g_out_count;

static int   g_frame_count;
static short g_last_in_8k;

/* High-shelf EQ state: fc=1500Hz gain=12dB fs=16000Hz */
static float g_eq_x1, g_eq_x2, g_eq_y1, g_eq_y2;

void noise_init(void) {
    if (!g_win_inited) {
        for (int i = 0; i < WIN_LEN; i++)
            sqrt_hann_win[i] = sinf(3.14159265358979323846f * (float)i / (float)(WIN_LEN - 1));
        g_win_inited = 1;
    }
    gtcrn_state_init(&g_state);
    g_fft_fwd = kiss_fftr_alloc(N_FFT, 0, NULL, NULL);
    g_fft_inv = kiss_fftr_alloc(N_FFT, 1, NULL, NULL);
    g_fifo_wpos   = 0;
    g_fifo_count  = 0;
    g_ola_pos     = 0;
    g_out_rpos    = 0;
    g_out_count   = 0;
    g_frame_count = 0;
    g_last_in_8k  = 0;
    memset(g_fifo,     0, sizeof(g_fifo));
    memset(g_ola,      0, sizeof(g_ola));
    memset(g_out_fifo, 0, sizeof(g_out_fifo));
}

void noise_deinit(void) {
    if (g_fft_fwd) { free(g_fft_fwd); g_fft_fwd = NULL; }
    if (g_fft_inv) { free(g_fft_inv); g_fft_inv = NULL; }
}

void noise_reduction(short *voiceIn, short *voiceOut) {
    /* ── 1. 8kHz → 16kHz 上采样 ── */
    for (int i = 0; i < FRAME_IN; i++) {
        float s_curr = (float)voiceIn[i] / 32768.0f;
        float s_prev = (float)g_last_in_8k / 32768.0f;
        g_fifo[g_fifo_wpos] = s_curr;
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo[g_fifo_wpos] = (s_prev + s_curr) * 0.5f;
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo_count += 2;
        g_last_in_8k = voiceIn[i];
    }

    /* ── 2. STFT→GTCRN→ISTFT (时间正序: 最旧→最新) ── */
    {
        int initial_fc = g_fifo_count;
        int read_wpos = (g_fifo_wpos - initial_fc + WIN_LEN + FIFO_SZ) % FIFO_SZ;
        while (g_fifo_count >= WIN_LEN) {
            g_fifo_count -= WIN_INC;

            kiss_fft_scalar fft_in[WIN_LEN];
            int start = (read_wpos - WIN_LEN + FIFO_SZ) % FIFO_SZ;
            for (int i = 0; i < WIN_LEN; i++)
                fft_in[i] = g_fifo[(start + i) % FIFO_SZ] * sqrt_hann_win[i];

            kiss_fft_cpx fft_out[N_BINS];
            kiss_fftr(g_fft_fwd, fft_in, fft_out);
            float real[N_BINS], imag[N_BINS];
            for (int i = 0; i < N_BINS; i++) {
                real[i] = fft_out[i].r;
                imag[i] = fft_out[i].i;
            }

            int32_t crm[2 * N_BINS];
            gtcrn_infer_frame(real, imag, &g_state,
                              erb_erb_fc_weight, erb_ierb_fc_weight, crm);

            kiss_fft_cpx inv_in[N_BINS];
            for (int i = 0; i < N_BINS; i++) {
                inv_in[i].r = (float)crm[i] / 1048576.0f;
                inv_in[i].i = (float)crm[N_BINS + i] / 1048576.0f;
            }
            kiss_fft_scalar ifft_out[WIN_LEN];
            kiss_fftri(g_fft_inv, inv_in, ifft_out);

            for (int i = 0; i < WIN_LEN; i++) {
                float sample = ifft_out[i] * sqrt_hann_win[i] / (float)N_FFT;
                int pos = (g_ola_pos + i) % (WIN_LEN + WIN_INC);
                g_ola[pos] += sample;
            }

            for (int i = 0; i < WIN_INC; i++) {
                float v = g_ola[g_ola_pos];
                g_ola[g_ola_pos] = 0.0f;
                g_ola_pos = (g_ola_pos + 1) % (WIN_LEN + WIN_INC);
                if (v >  1.0f) v =  1.0f;
                if (v < -1.0f) v = -1.0f;
                g_out_fifo[(g_out_rpos + g_out_count) % FIFO_SZ] = v;
                g_out_count++;
            }
            g_frame_count++;
            read_wpos = (read_wpos + WIN_INC) % FIFO_SZ;
        }
    }

    /* ── 3. 输出 8kHz PCM ── */
    if (g_out_count >= FRAME_16K) {
        float gain = 1.0f;
        if (g_frame_count < WARMUP_MUTE) {
            gain = 0.0f;
        } else if (g_frame_count < WARMUP_MUTE + WARMUP_FADE) {
            gain = (float)(g_frame_count - WARMUP_MUTE) / (float)WARMUP_FADE;
        }

        /* ═══════════════════════════════════════════════════════
         *  High-shelf EQ: fc=1300Hz +18dB @ 16kHz
         *  提升高频纠正 sqrt-Hann 频谱倾斜 (~14dB), 低频平坦
         * ═══════════════════════════════════════════════════════ */
        #define HISH_B0 5.454908f
        #define HISH_B1 -8.558202f
        #define HISH_B2 3.527070f
        #define HISH_A1 -0.892201f
        #define HISH_A2 0.315977f

        short out_16k[FRAME_16K];
        for (int i = 0; i < FRAME_16K; i++) {
            float x = g_out_fifo[g_out_rpos] * gain;
            g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;

            /* Biquad high-shelf */
            float y = HISH_B0*x + HISH_B1*g_eq_x1 + HISH_B2*g_eq_x2
                                 - HISH_A1*g_eq_y1 - HISH_A2*g_eq_y2;
            g_eq_x2 = g_eq_x1; g_eq_x1 = x;
            g_eq_y2 = g_eq_y1; g_eq_y1 = y;

            int iv = (int)(y * 32768.0f + (y >= 0 ? 0.5f : -0.5f));
            if (iv >  32767) iv =  32767;
            if (iv < -32768) iv = -32768;
            out_16k[i] = (short)iv;
        }
        g_out_count -= FRAME_16K;

        for (int i = 0; i < FRAME_IN; i++)
            voiceOut[i] = (short)(((int)out_16k[i*2] + (int)out_16k[i*2+1]) >> 1);
    } else {
        memset(voiceOut, 0, FRAME_IN * sizeof(short));
    }
}
