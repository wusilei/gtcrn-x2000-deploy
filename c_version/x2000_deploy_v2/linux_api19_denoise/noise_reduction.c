#include "noise_reduction.h"
#include "denoise_fp.h"
#include "denoise_matlab_weights.h"
#include "fft_q15.h"
#include "kiss_fftr.h"
#include <stdlib.h>
#include <string.h>

#define N_FFT 512
#define WIN_LEN 512
#define WIN_INC 256
#define N_BINS 257
#define FRAME_IN 200
#define FRAME_16K 400
#define FIFO_SZ 2048
#define WARMUP_MUTE 20
#define WARMUP_FADE 12

static float   hann_f[WIN_LEN], hann_div_n[WIN_LEN];
static int     g_win_inited = 0;

static denoise_state_t g_state;
static kiss_fftr_cfg  g_fft_fwd;
static float g_fifo[FIFO_SZ];
static int   g_fifo_wpos, g_fifo_count;
static float g_ola[WIN_LEN + WIN_INC];
static int   g_ola_pos;
static float g_out_fifo[FIFO_SZ];
static int   g_out_rpos, g_out_count;
static int   g_frame_count;
static short g_last_in_8k;

#define PCM_SCALE (1.0f/32768.0f)

void noise_init(void) {
    if (!g_win_inited) {
        for (int i = 0; i < WIN_LEN; i++) {
            float w = sinf(3.14159265358979323846f * (float)i / (float)(WIN_LEN - 1));
            hann_f[i] = w;
            hann_div_n[i] = w / (float)N_FFT;
        }
        g_win_inited = 1;
    }
    denoise_state_init(&g_state);
    g_fft_fwd = kiss_fftr_alloc(N_FFT, 0, NULL, NULL);
    g_fifo_wpos = g_fifo_count = 0; g_ola_pos = 0;
    g_out_rpos = g_out_count = 0; g_frame_count = 0; g_last_in_8k = 0;
    memset(g_fifo, 0, sizeof(g_fifo)); memset(g_ola, 0, sizeof(g_ola));
    memset(g_out_fifo, 0, sizeof(g_out_fifo));
}
void noise_deinit(void) { if (g_fft_fwd) { free(g_fft_fwd); g_fft_fwd = NULL; } }

void noise_reduction(short *voiceIn, short *voiceOut) {
    /* 1. upsample */
    for (int i = 0; i < FRAME_IN; i++) {
        float s_curr = (float)voiceIn[i] * PCM_SCALE;
        float s_prev = (float)g_last_in_8k * PCM_SCALE;
        g_fifo[g_fifo_wpos] = s_curr;
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo[g_fifo_wpos] = (s_prev + s_curr) * 0.5f;
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo_count += 2;
        g_last_in_8k = voiceIn[i];
    }

    /* 2. STFT→DENOISE→ISTFT */
    {
        int read_wpos = (g_fifo_wpos - g_fifo_count + WIN_LEN + FIFO_SZ) % FIFO_SZ;
        while (g_fifo_count >= WIN_LEN) {
            g_fifo_count -= WIN_INC;

            kiss_fft_scalar fft_in[WIN_LEN];
            int start = (read_wpos - WIN_LEN + FIFO_SZ) % FIFO_SZ;
            for (int i = 0; i < WIN_LEN; i++)
                fft_in[i] = g_fifo[(start + i) % FIFO_SZ] * hann_f[i];

            kiss_fft_cpx fft_out[N_BINS];
            kiss_fftr(g_fft_fwd, fft_in, fft_out);

            float real[N_BINS], imag[N_BINS];
            for (int i = 0; i < N_BINS; i++) {
                real[i] = fft_out[i].r;
                imag[i] = fft_out[i].i;
            }

            int32_t crm[2 * N_BINS];
            denoise_infer_frame(real, imag, &g_state,
                              erb_erb_fc_weight, erb_ierb_fc_weight, crm);

            /* Q15 IFFT */
            int32_t inv_r[N_BINS], inv_i[N_BINS];
            for (int i = 0; i < N_BINS; i++) {
                inv_r[i] = (crm[i] + 16) >> 5;
                inv_i[i] = (crm[N_BINS + i] + 16) >> 5;
            }
            int32_t ifft_q[WIN_LEN];
            fft_q15_inverse(inv_r, inv_i, ifft_q);

            for (int i = 0; i < WIN_LEN; i++) {
                float s = (float)ifft_q[i] * (1.0f / 32768.0f) * hann_div_n[i];
                int pos = (g_ola_pos + i) % (WIN_LEN + WIN_INC);
                g_ola[pos] += s;
            }

            for (int i = 0; i < WIN_INC; i++) {
                float v = g_ola[g_ola_pos];
                g_ola[g_ola_pos] = 0.0f;
                g_ola_pos = (g_ola_pos + 1) % (WIN_LEN + WIN_INC);
                if (v > 1.0f) v = 1.0f; if (v < -1.0f) v = -1.0f;
                g_out_fifo[(g_out_rpos + g_out_count) % FIFO_SZ] = v;
                g_out_count++;
            }
            g_frame_count++;
            read_wpos = (read_wpos + WIN_INC) % FIFO_SZ;
        }
    }

    /* 3. Output */
    if (g_out_count >= FRAME_16K) {
        float gain = 1.0f;
        if (g_frame_count < WARMUP_MUTE) gain = 0.0f;
        else if (g_frame_count < WARMUP_MUTE + WARMUP_FADE)
            gain = (float)(g_frame_count - WARMUP_MUTE) / (float)WARMUP_FADE;

        for (int i = 0; i < FRAME_IN; i++) {
            float a = g_out_fifo[g_out_rpos] * gain; g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;
            float b = g_out_fifo[g_out_rpos] * gain; g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;
            int iv = (int)((a + b) * 0.5f * 32768.0f + 0.5f);
            if (iv > 32767) iv = 32767; if (iv < -32768) iv = -32768;
            voiceOut[i] = (short)iv;
        }
        g_out_count -= FRAME_16K;
    } else {
        memset(voiceOut, 0, FRAME_IN * sizeof(short));
    }
}
