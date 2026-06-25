/**
 * noise_reduction.c — GTCRN Linux 侧降噪适配层 (流式, 参考 libgtcrn_stream)
 * ============================================================
 * 参考 /home/a/work/magik_deploy/x2000_deploy/libgtcrn_stream_q15/
 *
 * 关键: FIFO 取帧无 mirror padding, 标准 Hann 窗 + /wsum 归一化
 *
 * 8kHz PCM → 2×上采样 → 16kHz FIFO → STFT(Hann窗,无mirror) → GTCRN
 *   → IFFT ÷N_FFT ×Hann → OLA + /wsum → 2×下采样 → 8kHz PCM
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
#define FRAME_IN    200     /* 8kHz input  */
#define FRAME_16K   400     /* 16kHz after upsample */
#define FIFO_SZ     (WIN_LEN * 4)  /* 2048 sample FIFO */

/* Hann window (matching MATLAB win_hann.mat, NOT sqrt-Hann) */
static const float hann_win[WIN_LEN] = {
    0.000000f,0.006135f,0.012272f,0.018407f,0.024541f,0.030674f,0.036807f,
    0.042938f,0.049068f,0.055195f,0.061321f,0.067444f,0.073565f,0.079682f,
    0.085797f,0.091909f,0.098017f,0.104122f,0.110222f,0.116319f,0.122411f,
    0.128498f,0.134581f,0.140658f,0.146731f,0.152797f,0.158858f,0.164913f,
    0.170962f,0.177004f,0.183040f,0.189069f,0.195090f,0.201105f,0.207111f,
    0.213110f,0.219101f,0.225084f,0.231058f,0.237024f,0.242980f,0.248928f,
    0.254866f,0.260794f,0.266713f,0.272621f,0.278520f,0.284408f,0.290285f,
    0.296151f,0.302006f,0.307850f,0.313682f,0.319502f,0.325310f,0.331106f,
    0.336890f,0.342661f,0.348419f,0.354164f,0.359895f,0.365613f,0.371317f,
    0.377008f,0.382683f,0.388345f,0.393992f,0.399624f,0.405241f,0.410843f,
    0.416430f,0.422000f,0.427555f,0.433094f,0.438616f,0.444122f,0.449611f,
    0.455084f,0.460539f,0.465977f,0.471397f,0.476799f,0.482184f,0.487550f,
    0.492898f,0.498228f,0.503538f,0.508830f,0.514103f,0.519356f,0.524590f,
    0.529804f,0.534998f,0.540172f,0.545325f,0.550458f,0.555570f,0.560662f,
    0.565732f,0.570781f,0.575808f,0.580814f,0.585798f,0.590760f,0.595699f,
    0.600616f,0.605511f,0.610383f,0.615232f,0.620057f,0.624860f,0.629638f,
    0.634393f,0.639124f,0.643832f,0.648514f,0.653173f,0.657807f,0.662416f,
    0.667000f,0.671559f,0.676093f,0.680601f,0.685084f,0.689541f,0.693972f,
    0.698376f,0.702755f,0.707107f,0.711432f,0.715731f,0.720003f,0.724247f,
    0.728464f,0.732654f,0.736817f,0.740951f,0.745058f,0.749136f,0.753187f,
    0.757209f,0.761203f,0.765167f,0.769103f,0.773011f,0.776888f,0.780737f,
    0.784557f,0.788346f,0.792107f,0.795837f,0.799537f,0.803208f,0.806848f,
    0.810457f,0.814036f,0.817585f,0.821103f,0.824589f,0.828045f,0.831470f,
    0.834863f,0.838225f,0.841555f,0.844854f,0.848120f,0.851355f,0.854558f,
    0.857729f,0.860867f,0.863973f,0.867046f,0.870087f,0.873095f,0.876070f,
    0.879012f,0.881921f,0.884797f,0.887640f,0.890449f,0.893224f,0.895966f,
    0.898674f,0.901349f,0.903989f,0.906596f,0.909168f,0.911706f,0.914210f,
    0.916679f,0.919114f,0.921514f,0.923880f,0.926210f,0.928506f,0.930767f,
    0.932993f,0.935184f,0.937339f,0.939459f,0.941544f,0.943593f,0.945607f,
    0.947586f,0.949528f,0.951435f,0.953306f,0.955141f,0.956940f,0.958703f,
    0.960431f,0.962121f,0.963776f,0.965394f,0.966976f,0.968522f,0.970031f,
    0.971504f,0.972940f,0.974339f,0.975702f,0.977028f,0.978317f,0.979570f,
    0.980785f,0.981964f,0.983105f,0.984210f,0.985278f,0.986308f,0.987301f,
    0.988258f,0.989177f,0.990058f,0.990903f,0.991710f,0.992480f,0.993212f,
    0.993907f,0.994565f,0.995185f,0.995767f,0.996313f,0.996820f,0.997290f,
    0.997723f,0.998118f,0.998476f,0.998795f,0.999078f,0.999322f,0.999529f,
    0.999699f,0.999831f,0.999925f,0.999981f,1.000000f
};

/* ═══════════════════════════════════════════════════════════
 *  全部流式状态 (static)
 * ═══════════════════════════════════════════════════════════ */
static gtcrn_state_t g_state;
static kiss_fftr_cfg  g_fft_fwd, g_fft_inv;

/* 16kHz PCM FIFO */
static float g_fifo[FIFO_SZ];
static int   g_fifo_wpos;    /* write position */
static int   g_fifo_count;   /* valid samples in FIFO */

/* OLA buffer */
static float g_ola[WIN_LEN + WIN_INC];
static float g_wsum[WIN_LEN + WIN_INC];
static int   g_ola_pos;

/* Output FIFO (16kHz float) */
static float g_out_fifo[FIFO_SZ];
static int   g_out_rpos;
static int   g_out_count;

static int g_frame_count;

/* ═══════════════════════════════════════════════════════════
 *  noise_init
 * ═══════════════════════════════════════════════════════════ */
void noise_init(void) {
    gtcrn_state_init(&g_state);
    g_fft_fwd = kiss_fftr_alloc(N_FFT, 0, NULL, NULL);
    g_fft_inv = kiss_fftr_alloc(N_FFT, 1, NULL, NULL);
    g_fifo_wpos   = 0;
    g_fifo_count  = 0;
    g_ola_pos     = 0;
    g_out_rpos    = 0;
    g_out_count   = 0;
    g_frame_count = 0;
    memset(g_fifo,  0, sizeof(g_fifo));
    memset(g_ola,   0, sizeof(g_ola));
    memset(g_wsum,  0, sizeof(g_wsum));
    memset(g_out_fifo, 0, sizeof(g_out_fifo));
}

void noise_deinit(void) {
    if (g_fft_fwd) { free(g_fft_fwd); g_fft_fwd = NULL; }
    if (g_fft_inv) { free(g_fft_inv); g_fft_inv = NULL; }
}

/* ═══════════════════════════════════════════════════════════
 *  noise_reduction — 参考 libgtcrn_stream 架构
 * ═══════════════════════════════════════════════════════════ */
void noise_reduction(short *voiceIn, short *voiceOut) {
    /* ── 1. 8kHz → 16kHz 上采样, 写入 FIFO ── */
    for (int i = 0; i < FRAME_IN - 1; i++) {
        float s0 = (float)voiceIn[i] / 32768.0f;
        float s1 = (float)voiceIn[i+1] / 32768.0f;
        g_fifo[g_fifo_wpos] = s0;
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo[g_fifo_wpos] = (s0 + s1) * 0.5f;
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo_count += 2;
    }
    {
        float s = (float)voiceIn[FRAME_IN-1] / 32768.0f;
        g_fifo[g_fifo_wpos] = s;
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo[g_fifo_wpos] = s;
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo_count += 2;
    }

    /* ── 2. 每凑够 HOP 样本, 取一帧做 STFT→GTCRN→ISTFT ── */
    while (g_fifo_count >= WIN_INC) {
        g_fifo_count -= WIN_INC;

        /* 从 FIFO 取最后 WIN_LEN 个样本 (参考 libgtcrn_stream: 无 mirror padding) */
        kiss_fft_scalar fft_in[WIN_LEN];
        int start = (g_fifo_wpos - WIN_LEN + FIFO_SZ) % FIFO_SZ;
        for (int i = 0; i < WIN_LEN; i++)
            fft_in[i] = g_fifo[(start + i) % FIFO_SZ] * hann_win[i];

        /* FFT */
        kiss_fft_cpx fft_out[N_BINS];
        kiss_fftr(g_fft_fwd, fft_in, fft_out);
        float real[N_BINS], imag[N_BINS];
        for (int i = 0; i < N_BINS; i++) {
            real[i] = fft_out[i].r;
            imag[i] = fft_out[i].i;
        }

        /* GTCRN 推理 */
        int32_t crm[2 * N_BINS];
        gtcrn_infer_frame(real, imag, &g_state,
                          erb_erb_fc_weight, erb_ierb_fc_weight, crm);

        /* IFFT */
        kiss_fft_cpx inv_in[N_BINS];
        for (int i = 0; i < N_BINS; i++) {
            inv_in[i].r = (float)crm[i] / 1048576.0f;
            inv_in[i].i = (float)crm[N_BINS + i] / 1048576.0f;
        }
        kiss_fft_scalar ifft_out[WIN_LEN];
        kiss_fftri(g_fft_inv, inv_in, ifft_out);

        /* OLA: ifft_out / N_FFT × hann_win → accumulate → /wsum */
        for (int i = 0; i < WIN_LEN; i++) {
            float sample = ifft_out[i] * hann_win[i] / (float)N_FFT;
            int pos = (g_ola_pos + i) % (WIN_LEN + WIN_INC);
            g_ola[pos]  += sample;
            g_wsum[pos] += hann_win[i] * hann_win[i];
        }

        /* 输出 WIN_INC 个样本 */
        for (int i = 0; i < WIN_INC; i++) {
            float v = 0.0f;
            if (g_wsum[g_ola_pos] > 1e-8f)
                v = g_ola[g_ola_pos] / g_wsum[g_ola_pos];
            g_ola[g_ola_pos]  = 0.0f;
            g_wsum[g_ola_pos] = 0.0f;
            g_ola_pos = (g_ola_pos + 1) % (WIN_LEN + WIN_INC);

            if (v >  1.0f) v =  1.0f;
            if (v < -1.0f) v = -1.0f;

            g_out_fifo[(g_out_rpos + g_out_count) % FIFO_SZ] = v;
            g_out_count++;
        }
        g_frame_count++;
    }

    /* ── 3. 输出 8kHz PCM ── */
    if (g_out_count >= FRAME_16K) {
        /* Warmup: first 40 frames (~1s) fade in */
        float gain = 1.0f;
        if (g_frame_count < 60) gain = (float)g_frame_count / 60.0f;

        short out_16k[FRAME_16K];
        for (int i = 0; i < FRAME_16K; i++) {
            float v = g_out_fifo[g_out_rpos] * 32768.0f * gain;
            g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;
            int iv = (int)(v + (v >= 0 ? 0.5f : -0.5f));
            if (iv >  32767) iv =  32767;
            if (iv < -32768) iv = -32768;
            out_16k[i] = (short)iv;
        }
        g_out_count -= FRAME_16K;

        /* 16k→8k 下采样 */
        for (int i = 0; i < FRAME_IN; i++)
            voiceOut[i] = (short)(((int)out_16k[i*2] + (int)out_16k[i*2+1]) >> 1);
    } else {
        memset(voiceOut, 0, FRAME_IN * sizeof(short));
    }
}
