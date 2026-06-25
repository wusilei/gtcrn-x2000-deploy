/**
 * gtcrn_fp.h — GTCRN MATLAB→C Fixed-Point Implementation
 * =======================================================
 * Q-format matching MATLAB Fix_point.m exactly:
 *
 *   s32f20: activations (int32_t, 20 frac bits, ×1048576)
 *   s32f18: Conv weights encoder (int32_t, 18 frac bits, ×262144)
 *   s16f15: GRU hidden / sigmoid output (int16_t, 15 frac bits, ×32768)
 *   s16f14: BN/PReLU/DeConv weights (int16_t, 14 frac bits, ×16384)
 *   s16f13: FC/DD-Conv/PC weights (int16_t, 13 frac bits, ×8192)
 *   s16f12: GRU/LN/DD-DeConv weights (int16_t, 12 frac bits, ×4096)
 *   s16f10: GRU bias (int16_t, 10 frac bits, ×1024)
 *   u16f15: sigmoid/tanh output (uint16_t, 15 frac bits, ×32768)
 *   u16f14: BN weight (uint16_t, 14 frac bits, ×16384)
 *   u16f13: BN running_var (uint16_t, 13 frac bits, ×8192)
 *   u16f12: BN running_var (uint16_t, 12 frac bits, ×4096)
 *   u16f10: BN running_var (uint16_t, 10 frac bits, ×1024)
 *
 * Target: Ingenic X2000 MIPS32R2 (no FPU) + PC verification
 *
 * Data layout: (C, W) for 2D tensors, (C, H, W) for 3D
 *   Indexing: element[c][h][w] = data[c*H*W + h*W + w]
 *   For (C, W): element[c][w] = data[c*W + w]
 */

#ifndef GTCRN_FP_H
#define GTCRN_FP_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Q-format Constants (matching MATLAB Fix_point.m)
 * ================================================================ */

/* Activation Q-formats */
#define Q_ACT       20    /* s32f20: main activations */
#define Q_ACT_TR    18    /* s32f18: TRA/DeTRA agg quant */

/* Weight Q-formats */
#define Q_WGT_CONV_E  18  /* s32f18: Conv weights (encoder Conv0, DeConv0) */
#define Q_WGT_FC      13  /* s16f13: FC weights, DD-Conv, PC weights */
#define Q_WGT_GRU     12  /* s16f12: GRU weights, LN weights, DD-DeConv */
#define Q_WGT_BN      14  /* s16f14 / u16f14: BN/PReLU weights */
#define Q_WGT_GRU_BIAS 10 /* s16f10: GRU bias */
#define Q_WGT_SIGMOID 15  /* u16f15: sigmoid/tanh output */
#define Q_WGT_BM_BS   15  /* u16f15: BM/BS weights */

/* BN running_var Q-formats (varies per block) */
#define Q_BN_VAR_14   14  /* u16f14 */
#define Q_BN_VAR_13   13  /* u16f13 */
#define Q_BN_VAR_12   12  /* u16f12 */
#define Q_BN_VAR_10   10  /* u16f10 */

/* Safe clamp to int16_t range (s16f15 max is 32767, not 32768) */
static inline int16_t sat_s16f15(float x) {
    float r = roundf(x * 32768.0f);
    if (r > 32767.0f) return 32767;
    if (r < -32768.0f) return -32768;
    return (int16_t)r;
}

/* Scale factors */
#define F2Q20(x)  ((int32_t)roundf((x) * 1048576.0f))   /* × 2^20 */
#define F2Q18(x)  ((int32_t)roundf((x) * 262144.0f))    /* × 2^18 */
#define F2Q15(x)  (sat_s16f15(x))                        /* × 2^15, safely clamped */
#define F2Q14(x)  ((int16_t)roundf((x) * 16384.0f))     /* × 2^14 */
#define F2Q13(x)  ((int16_t)roundf((x) * 8192.0f))      /* × 2^13 */
#define F2Q12(x)  ((int16_t)roundf((x) * 4096.0f))      /* × 2^12 */
#define F2Q10(x)  ((int16_t)roundf((x) * 1024.0f))      /* × 2^10 */

#define U2Q15(x)  ((uint16_t)roundf((x) * 32768.0f))
#define U2Q14(x)  ((uint16_t)roundf((x) * 16384.0f))
#define U2Q13(x)  ((uint16_t)roundf((x) * 8192.0f))
#define U2Q12(x)  ((uint16_t)roundf((x) * 4096.0f))
#define U2Q10(x)  ((uint16_t)roundf((x) * 1024.0f))

/* Dequant */
#define Q20_TO_F(x)  ((float)(x) / 1048576.0f)
#define Q18_TO_F(x)  ((float)(x) / 262144.0f)
#define Q15_TO_F(x)  ((float)(x) / 32768.0f)
#define Q10_TO_F(x)  ((float)(x) / 1024.0f)

/* ================================================================
 * Saturation Helpers
 * ================================================================ */

static inline int32_t sat32(int64_t x) {
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

static inline int16_t sat16(int32_t x) {
    if (x > INT16_MAX) return INT16_MAX;
    if (x < INT16_MIN) return INT16_MIN;
    return (int16_t)x;
}

static inline uint16_t usat16(int32_t x) {
    if (x > UINT16_MAX) return UINT16_MAX;
    if (x < 0) return 0;
    return (uint16_t)x;
}

/* ROUND(x) in C: add half and truncate towards zero */
static inline int32_t round_div(int64_t num, int64_t den) {
    int64_t half = den >> 1;
    if (num >= 0) return (int32_t)((num + half) / den);
    else return (int32_t)((num - half) / den);
}

/* ================================================================
 * Sigmoid & Tanh (soft-float, <5% total ops on X2000)
 * ================================================================ */

static inline float sigmoidf_fp(float x) {
    if (x > 88.0f) return 1.0f;
    if (x < -88.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static inline float tanhf_fp(float x) {
    return tanhf(x);
}

/* ================================================================
 * Dimension Constants
 * ================================================================ */

#define N_FFT       512
#define WIN_LEN     512
#define WIN_INC     256
#define N_BINS      257
#define N_BINS_BM   129
#define N_BINS_MID  65
#define N_BINS_SMALL 33

#define CH_IN       3     /* mag + real + imag */
#define CH_SFE      9     /* after SFE (3×3) */
#define CH_MID      16    /* intermediate channels */
#define CH_GRU_HID  4     /* Intra-RNN GRU hidden */
#define CH_GRU_INTER 8    /* Inter-RNN GRU hidden */
#define CH_TRA_HID  16    /* TRA GRU hidden */
#define CH_PC0_IN   24    /* PC0 input (8×3 from SFE) */
#define CH_OUT      2     /* final CRM output (I, Q) */

#define DILATION_MAX 5   /* max dilation for history buffer */

/* ================================================================
 * History / State Buffers (per-frame persistent state)
 * ================================================================ */

/* DD-Conv 3D history: (C, time_steps, freq)
 * Encoder: 16 channels × 16 time × 33 freq + current frame
 * Decoder: 16 channels × 16 time × 33 freq + current frame
 */
#define DD_HIST_TIME  16  /* history length = kernel_size + 2*(dil-1) - 1 = 3+2*(5-1)-1 = 14, round to 16 */
#define DD_HIST_SIZE  (CH_MID * DD_HIST_TIME * N_BINS_SMALL)

/* GRU hidden states */
#define GRU_HIDDEN_SIZE  (N_BINS_SMALL * CH_GRU_INTER)  /* 33 × 8 */

/* TRA hidden states: (N_BINS_SMALL, CH_TRA_HID) = 33×16 */
#define TRA_HIDDEN_SIZE  (N_BINS_SMALL * CH_TRA_HID)

/* ================================================================
 * Model State (extern — allocated by user)
 * ================================================================ */

typedef struct {
    /* Encoder DD-Conv history: (16, DD_HIST_TIME, 33) per GT-Conv ×3 */
    int32_t enc_conv_hist[CH_MID * DD_HIST_TIME * N_BINS_SMALL];

    /* Encoder TRA GRU hidden: (3 * 16) — 3 GT-Convs, single state each */
    int16_t enc_h_prev[3 * CH_TRA_HID];

    /* Decoder DD-DeConv history: (16, DD_HIST_TIME, 33) per GT-DeConv ×3 */
    int32_t dec_conv_hist[CH_MID * DD_HIST_TIME * N_BINS_SMALL];

    /* Decoder DeTRA GRU hidden: (3 * 16) — 3 GT-DeConvs, single state each */
    int16_t dec_h_prev[3 * CH_TRA_HID];

    /* Inter-RNN hidden: (2×DPRNN, 33, 16) — 2 DPRNN blocks */
    int16_t inter_prev1[N_BINS_SMALL * CH_MID];
    int16_t inter_prev2[N_BINS_SMALL * CH_MID];

} gtcrn_state_t;

/* ================================================================
 * Function Prototypes — Basic Ops
 * ================================================================ */

/* conv2d: (Cin, Win) → (Cout, Wout), weight (Cout, Cin, Hk, Wk) */
void conv2d_fixed(const int32_t *x, int Cin, int Win,
                  const int16_t *weight, const int32_t *bias,
                  int Cout, int Wout, int Hk, int Wk,
                  int stride, int pad_w, int qr,
                  int32_t *y);

/* tconv2d: transposed conv2d */
void tconv2d_fixed(const int32_t *x, int Cin, int Win,
                   const int16_t *weight, const int32_t *bias,
                   int Cout, int Wout, int Hk, int Wk,
                   int stride, int qr,
                   int32_t *y);

/* ddconv2d: depth-wise dilated conv2d, (C, H, W) input with history */
void ddconv2d_fixed(const int32_t *x, int C, int H, int Win,
                    const int16_t *weight, const int32_t *bias,
                    int Wout, int Hk, int Wk,
                    int pad_h, int pad_w, int dilation, int qr,
                    int32_t *y,
                    int32_t *hist_out, int *hist_len);

/* ddtconv2d: depth-wise dilated transposed conv2d */
void ddtconv2d_fixed(const int32_t *x, int C, int H, int Win,
                     const int16_t *weight, const int32_t *bias,
                     int Wout, int Hk, int Wk,
                     int pad_w, int dilation, int qr,
                     int32_t *y,
                     int32_t *hist_out, int *hist_len);

/* pconv2d: point-wise conv2d (1×1), weights (Cout, Cin, 1, 1) */
void pconv2d_fixed(const int32_t *x, int Cin, int Win,
                   const int16_t *weight, const int32_t *bias,
                   int Cout, int qr,
                   int32_t *y);

/* ptconv2d: point-wise transposed conv2d (1×1) */
void ptconv2d_fixed(const int32_t *x, int Cin, int Win,
                    const int16_t *weight, const int32_t *bias,
                    int Cout, int qr,
                    int32_t *y);

/* BatchNorm: y = ((x - mean) * var_inv)>>qr1 * weight>>qr2 + bias */
void bn_fixed(int32_t *x, int C, int Win,
              const uint16_t *weight, const int32_t *bias,
              const int32_t *running_mean, const uint16_t *running_var,
              int qr1, int qr2);

/* PReLU: y(x<0) = x * weight >> qr */
void prelu_fixed(int32_t *x, int C, int Win,
                 const int16_t *weight, int qr);

/* LayerNorm: computed online (mean/var from data) */
void ln_fixed(int32_t *x, int C, int Win,
              const int16_t *weight, const int32_t *bias, int qr);

/* GRU: (seq_len, input_dim) with s32f20 input → s16f15 output */
void gru_fixed(const int32_t *x, int seq_len, int input_dim, int hidden_dim,
               const int16_t *ih_weight, const int16_t *ih_bias,
               const int16_t *hh_weight, const int16_t *hh_bias,
               int16_t *y, int16_t *h_prev);

/* Per-frame independent GRU — each frame has its own hidden state at h_prev[f*hidden_dim] */
void gru_fixed_perframe(const int32_t *x, int seq_len, int input_dim, int hidden_dim,
                        const int16_t *ih_weight, const int16_t *ih_bias,
                        const int16_t *hh_weight, const int16_t *hh_bias,
                        int16_t *y, int16_t *h_prev,
                        int state_stride);

/* BiGRU: bidirectional GRU */
void bigru_fixed(const int32_t *x, int seq_len, int input_dim, int hidden_dim,
                 const int16_t *ih_w, const int16_t *ih_b,
                 const int16_t *hh_w, const int16_t *hh_b,
                 const int16_t *re_ih_w, const int16_t *re_ih_b,
                 const int16_t *re_hh_w, const int16_t *re_hh_b,
                 int16_t *y);

/* SFE: Subband Feature Extraction: (C, W) → (C*3, W) */
void SFE_fixed(const int32_t *x, int C, int Win, int32_t *y);

/* BM: Band Merging: (3, 257) → (3, 129) */
void BM_fixed(const int32_t *x, const uint16_t *weight, int32_t *y);

/* BS: Band Splitting: (2, 129) → (2, 257), input s16f15 → output s32f20 */
void BS_fixed(const int16_t *x, const uint16_t *weight, int16_t *y);

/* MASK: Complex Ratio Mask, mask s16f15 × real/imag s32f20 → output s32f20 */
void MASK_fixed(const int16_t *mask, const int32_t *real_in,
                const int32_t *imag_in, int32_t *y);

/* mag_gen: magnitude + real + imag → (3, 257) s32f20 */
void mag_gen(const float *real_in, const float *imag_in, int N, int32_t *y);

/* ================================================================
 * Function Prototypes — High-Level Modules
 * ================================================================ */

/* DD-Conv block: depthwise dilated conv + BN + PReLU */
void DD_Conv_block(const int32_t *x, int32_t *conv_hist,
                   int dilation, int gtc_idx,
                   const int16_t *conv_w, const int32_t *conv_b,
                   const uint16_t *bn_w, const int32_t *bn_b,
                   const int32_t *bn_mean, const uint16_t *bn_var,
                   const int16_t *prelu_w,
                   int32_t *y);

/* DD-DeConv block: depthwise dilated transposed conv + BN + PReLU */
void DD_DeConv_block(const int32_t *x, int32_t *conv_hist,
                     int dilation, int gtc_idx,
                     const int16_t *conv_w, const int32_t *conv_b,
                     const uint16_t *bn_w, const int32_t *bn_b,
                     const int32_t *bn_mean, const uint16_t *bn_var,
                     const int16_t *prelu_w,
                     int32_t *y);

/* TRA: Temporal Recurrent Attention */
void TRA_module(const int32_t *x, int16_t *h_prev, int gtc_idx,
                const int16_t *ih_w, const int32_t *ih_b,  /* s16f13, s32f20 */
                const int16_t *hh_w, const int32_t *hh_b,
                const int16_t *fc_w, const int32_t *fc_b,
                int32_t *y);

/* DeTRA: Decoder Temporal Recurrent Attention */
void DeTRA_module(const int32_t *x, int16_t *h_prev, int gtc_idx,
                  const int16_t *ih_w, const int32_t *ih_b,  /* s16f13, s32f18 */
                  const int16_t *hh_w, const int32_t *hh_b,
                  const int16_t *fc_w, const int32_t *fc_b,
                  int32_t *y);

/* GT-Conv: Grouped Temporal Convolution block */
void GT_Conv_module(const int32_t *x, int32_t *conv_hist, int16_t *h_prev,
                    int dilation, int gtc_idx,
                    const int16_t *pc0_w, const int32_t *pc0_b,
                    const uint16_t *pc0_bn_w, const int32_t *pc0_bn_b,
                    const int32_t *pc0_bn_m, const uint16_t *pc0_bn_v,
                    const int16_t *pc0_prelu_w,
                    const int16_t *dd_w, const int32_t *dd_b,
                    const uint16_t *dd_bn_w, const int32_t *dd_bn_b,
                    const int32_t *dd_bn_m, const uint16_t *dd_bn_v,
                    const int16_t *dd_prelu_w,
                    const int16_t *pc1_w, const int32_t *pc1_b,
                    const uint16_t *pc1_bn_w, const int32_t *pc1_bn_b,
                    const int32_t *pc1_bn_m, const uint16_t *pc1_bn_v,
                    const int16_t *tra_ih_w, const int32_t *tra_ih_b,
                    const int16_t *tra_hh_w, const int32_t *tra_hh_b,
                    const int16_t *tra_fc_w, const int32_t *tra_fc_b,
                    int32_t *y);

/* GT-DeConv: Grouped Temporal Transposed Convolution block */
void GT_DeConv_module(const int32_t *x_in, const int32_t *x_skip,
                      int32_t *conv_hist, int16_t *h_prev,
                      int dilation, int gtc_idx,
                      const int16_t *pc0_w, const int32_t *pc0_b,
                      const uint16_t *pc0_bn_w, const int32_t *pc0_bn_b,
                      const int32_t *pc0_bn_m, const uint16_t *pc0_bn_v,
                      const int16_t *pc0_prelu_w,
                      const int16_t *dd_w, const int32_t *dd_b,
                      const uint16_t *dd_bn_w, const int32_t *dd_bn_b,
                      const int32_t *dd_bn_m, const uint16_t *dd_bn_v,
                      const int16_t *dd_prelu_w,
                      const int16_t *pc1_w, const int32_t *pc1_b,
                      const uint16_t *pc1_bn_w, const int32_t *pc1_bn_b,
                      const int32_t *pc1_bn_m, const uint16_t *pc1_bn_v,
                      const int16_t *detra_ih_w, const int32_t *detra_ih_b,
                      const int16_t *detra_hh_w, const int32_t *detra_hh_b,
                      const int16_t *detra_fc_w, const int32_t *detra_fc_b,
                      int32_t *y);

/* Conv_block_0: standard conv2d → BN → PReLU */
void Conv_block_0(const int32_t *x,
                  const int32_t *conv_w, const int32_t *conv_b,
                  const uint16_t *bn_w, const int32_t *bn_b,
                  const int32_t *bn_m, const uint16_t *bn_v,
                  const int16_t *prelu_w,
                  int32_t *y);

/* Conv_block_1: grouped conv2d → BN → PReLU */
void Conv_block_1(const int32_t *x,
                  const int16_t *conv_w, const int32_t *conv_b,
                  const uint16_t *bn_w, const int32_t *bn_b,
                  const int32_t *bn_m, const uint16_t *bn_v,
                  const int16_t *prelu_w,
                  int32_t *y);

/* DeConv_block_0: transposed conv2d + skip → BN → Tanh */
void DeConv_block_0(const int32_t *x_in, const int32_t *x_skip,
                    const int32_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    int16_t *y);

/* DeConv_block_1: grouped transposed conv2d + skip → BN → PReLU */
void DeConv_block_1(const int32_t *x_in, const int32_t *x_skip,
                    const int16_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    const int16_t *prelu_w,
                    int32_t *y);

/* P_Conv_block_0: pointwise conv (Cin=24, Cout=16) → BN → PReLU */
void P_Conv_block_0(const int32_t *x, int gtc_idx,
                    const int16_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    const int16_t *prelu_w,
                    int32_t *y);

/* P_Conv_block_1: pointwise conv (Cin=16, Cout=8) → BN */
void P_Conv_block_1(const int32_t *x, int gtc_idx,
                    const int16_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    int32_t *y);

/* P_DeConv_block_0: pointwise transposed conv (Cin=24, Cout=16) → BN → PReLU */
void P_DeConv_block_0(const int32_t *x, int gtc_idx,
                      const int16_t *conv_w, const int32_t *conv_b,
                      const uint16_t *bn_w, const int32_t *bn_b,
                      const int32_t *bn_m, const uint16_t *bn_v,
                      const int16_t *prelu_w,
                      int32_t *y);

/* P_DeConv_block_1: pointwise transposed conv (Cin=16, Cout=8) → BN */
void P_DeConv_block_1(const int32_t *x, int gtc_idx,
                      const int16_t *conv_w, const int32_t *conv_b,
                      const uint16_t *bn_w, const int32_t *bn_b,
                      const int32_t *bn_m, const uint16_t *bn_v,
                      int32_t *y);

/* Intra_RNN: BiGRU + FC + LN + residual */
void Intra_RNN_module(const int32_t *x, int dprnn_idx,
                      const int16_t *rnn1_ih_w, const int16_t *rnn1_ih_b,
                      const int16_t *rnn1_hh_w, const int16_t *rnn1_hh_b,
                      const int16_t *rnn1_re_ih_w, const int16_t *rnn1_re_ih_b,
                      const int16_t *rnn1_re_hh_w, const int16_t *rnn1_re_hh_b,
                      const int16_t *rnn2_ih_w, const int16_t *rnn2_ih_b,
                      const int16_t *rnn2_hh_w, const int16_t *rnn2_hh_b,
                      const int16_t *rnn2_re_ih_w, const int16_t *rnn2_re_ih_b,
                      const int16_t *rnn2_re_hh_w, const int16_t *rnn2_re_hh_b,
                      const int16_t *fc_w, const int32_t *fc_b,
                      const int16_t *ln_w, const int32_t *ln_b,
                      int32_t *y);

/* Inter_RNN: GRU + FC + LN + residual */
void Inter_RNN_module(const int32_t *x, int16_t *h_prev, int dprnn_idx,
                      const int16_t *rnn1_ih_w, const int16_t *rnn1_ih_b,
                      const int16_t *rnn1_hh_w, const int16_t *rnn1_hh_b,
                      const int16_t *rnn2_ih_w, const int16_t *rnn2_ih_b,
                      const int16_t *rnn2_hh_w, const int16_t *rnn2_hh_b,
                      const int16_t *fc_w, const int32_t *fc_b,
                      const int16_t *ln_w, const int32_t *ln_b,
                      int32_t *y);

/* GDPRNN: Intra-RNN → Inter-RNN */
void GDPRNN_module(const int32_t *x, int16_t *inter_prev, int dprnn_idx,
                   const int16_t *rnn1_ih_w, const int16_t *rnn1_ih_b,
                   const int16_t *rnn1_hh_w, const int16_t *rnn1_hh_b,
                   const int16_t *rnn1_re_ih_w, const int16_t *rnn1_re_ih_b,
                   const int16_t *rnn1_re_hh_w, const int16_t *rnn1_re_hh_b,
                   const int16_t *rnn2_ih_w, const int16_t *rnn2_ih_b,
                   const int16_t *rnn2_hh_w, const int16_t *rnn2_hh_b,
                   const int16_t *rnn2_re_ih_w, const int16_t *rnn2_re_ih_b,
                   const int16_t *rnn2_re_hh_w, const int16_t *rnn2_re_hh_b,
                   const int16_t *inter_rnn1_ih_w, const int16_t *inter_rnn1_ih_b,
                   const int16_t *inter_rnn1_hh_w, const int16_t *inter_rnn1_hh_b,
                   const int16_t *inter_rnn2_ih_w, const int16_t *inter_rnn2_ih_b,
                   const int16_t *inter_rnn2_hh_w, const int16_t *inter_rnn2_hh_b,
                   const int16_t *intra_fc_w, const int32_t *intra_fc_b,
                   const int16_t *intra_ln_w, const int32_t *intra_ln_b,
                   const int16_t *inter_fc_w, const int32_t *inter_fc_b,
                   const int16_t *inter_ln_w, const int32_t *inter_ln_b,
                   int32_t *y);

/* Encoder: Conv0 → Conv1 → GT-Conv(×3) */
void Encoder_module(const int32_t *x, int32_t *conv_hist, int16_t *h_prev,
                    int32_t *y0, int32_t *y1, int32_t *y2,
                    int32_t *y3, int32_t *y4);

/* Decoder: GT-DeConv(×3) → DeConv1 → DeConv0 */
void Decoder_module(const int32_t *x, const int32_t *y0, const int32_t *y1,
                    const int32_t *y2, const int32_t *y3, const int32_t *y4,
                    int32_t *conv_hist, int16_t *h_prev,
                    int32_t *y);

/* Main inference: single-frame processing, CRM output s32f20 */
void gtcrn_infer_frame(const float *real_in, const float *imag_in,
                       gtcrn_state_t *state,
                       const uint16_t *erbfc_w, const uint16_t *ierbfc_w,
                       int32_t *crm_out);

/* Initialize state to zero */
void gtcrn_state_init(gtcrn_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_FP_H */
