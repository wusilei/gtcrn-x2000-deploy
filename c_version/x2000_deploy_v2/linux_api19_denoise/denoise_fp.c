/**
 * denoise_fp.c — DENOISE MATLAB→C Fixed-Point Implementation
 * =======================================================
 * Verbatim MATLAB→C translation of DENOISE_speech_enhance_FPversion/.
 *
 * Q-format:
 *   Activations:  s32f20 (int32_t, ×2^20)
 *   GRU hidden:   s16f15 (int16_t, ×2^15)
 *   Conv weights: s32f18 or s16f13 or s16f12 (block-dependent)
 *   BN weights:   u16f14 + u16f1x for var
 *   PReLU:        s16f14
 *   FC weights:   s16f13
 *   LN/GRU:       s16f12
 *   GRU bias:     s16f10
 *
 * Target: Ingenic X2000 MIPS32R2 (no FPU) + PC verification.
 * Soft-float calls (sigmoid/tanh) < 5% of total ops.
 */

#include "denoise_fp.h"

/* Direct LUT dispatch — eliminates float round-trip in sigmoid/tanh */
#define Q10_SIGMOID(x)  sigmoid_q15(x)
#define Q10_TANH(x)     tanh_q15(x)
#define Q20_SIGMOID(x)  sigmoid_q15((int32_t)(((int64_t)(x) + 512) >> 10))
#define Q20_TANH(x)     tanh_q15((int32_t)(((int64_t)(x) + 512) >> 10))
#define Q18_SIGMOID(x)  sigmoid_q15((int32_t)(((int64_t)(x) + 128) >> 8))
#define Q18_TANH(x)     tanh_q15((int32_t)(((int64_t)(x) + 128) >> 8))

/* ================================================================
 * Basic Ops — conv2d_fixed
 * ================================================================
 * Matches conv2d_func.m exactly.
 * x: (Cin, Win) in s32f20
 * weight: (Cout, Cin, Hk, Wk) in given Q-format
 * bias: (Cout,) in s32f20
 * y: (Cout, Wout) in s32f20
 */

void conv2d_fixed(const int32_t *x, int Cin, int Win,
                  const int16_t *weight, const int32_t *bias,
                  int Cout, int Wout, int Hk, int Wk,
                  int stride, int pad_w, int qr,
                  int32_t *y) {
    int Hout = 1;

    for (int co = 0; co < Cout; co++) {
        for (int w = 0; w < Wout; w++) {
            y[co * Wout + w] = bias[co];
        }

        for (int ci = 0; ci < Cin; ci++) {
            const int32_t *x_chan = x + ci * Win;

            for (int wo = 0; wo < Wout; wo++) {
                int64_t acc = 0;

                for (int hk = 0; hk < Hk; hk++) {
                    for (int wk = 0; wk < Wk; wk++) {
                        int wi = wo * stride + wk - pad_w;
                        int32_t x_val = 0;
                        if (wi >= 0 && wi < Win) x_val = x_chan[wi];
                        int kidx = ((co * Cin + ci) * Hk + hk) * Wk + wk;
                        int16_t k_val = weight[kidx];
                        int64_t prod = (int64_t)x_val * (int64_t)k_val;
                        int shift = -qr;
                        prod = (prod + ((int64_t)1 << (shift - 1))) >> shift;
                        acc += prod;
                    }
                }
                y[co * Wout + wo] = sat32((int64_t)y[co * Wout + wo] + acc);
            }
        }
    }
}

/* ================================================================
 * tconv2d_fixed — Transposed Conv2D
 * ================================================================
 * weight: (Cin, Cout, Hk, Wk) — (in, out) ordering
 * Kernel pre-rotation: rot90(kernel.', 2)
 */

void tconv2d_fixed(const int32_t *x, int Cin, int Win,
                   const int16_t *weight, const int32_t *bias,
                   int Cout, int Wout, int Hk, int Wk,
                   int stride, int qr,
                   int32_t *y) {
    int W_insert = Win + (Win - 1) * (stride - 1);
    int pad_w = 2;
    int W_padded = W_insert + 2 * pad_w;

    for (int co = 0; co < Cout; co++) {
        for (int w = 0; w < Wout; w++) y[co * Wout + w] = bias[co];

        for (int ci = 0; ci < Cin; ci++) {
            const int32_t *x_chan = x + ci * Win;
            int32_t *x_insert = (int32_t *)calloc(W_padded, sizeof(int32_t));
            for (int w = 0; w < Win; w++) x_insert[pad_w + w * stride] = x_chan[w];

            for (int wo = 0; wo < Wout; wo++) {
                int64_t acc = 0;
                for (int hk = 0; hk < Hk; hk++) {
                    for (int wk = 0; wk < Wk; wk++) {
                        int wi = wo + wk;
                        int32_t xv = (wi >= 0 && wi < W_padded) ? x_insert[wi] : 0;
                        int wk_rev = Wk - 1 - wk;
                        int kidx = ((ci * Cout + co) * Hk + hk) * Wk + wk_rev;
                        int16_t kv = weight[kidx];
                        int64_t prod = (int64_t)xv * (int64_t)kv;
                        int shift = -qr;
                        prod = (prod + ((int64_t)1 << (shift - 1))) >> shift;
                        acc += prod;
                    }
                }
                y[co * Wout + wo] = sat32((int64_t)y[co * Wout + wo] + acc);
            }
            free(x_insert);
        }
    }
}

/* ================================================================
 * ddconv2d_fixed — Depth-wise Dilated Conv2D with History
 * ================================================================
 * x: (C, H_in, Win), weight: (Cout, 1, Hk, Wk) s16f13
 */

void ddconv2d_fixed(const int32_t *x, int C, int H_in, int Win,
                    const int16_t *weight, const int32_t *bias,
                    int Wout, int Hk, int Wk,
                    int pad_h, int pad_w, int dilation, int qr,
                    int32_t *y,
                    int32_t *hist_out, int *hist_len) {
    int Cout = C;
    int Hout = 1;
    int Hk_dila = Hk + 2 * (dilation - 1);

    for (int chan = 0; chan < Cout; chan++) {
        int H_full = *hist_len + H_in;
        int32_t *x_full = (int32_t *)calloc(H_full * Win, sizeof(int32_t));
        int hist_start = chan * (*hist_len) * Win;
        for (int h = 0; h < *hist_len; h++)
            for (int w = 0; w < Win; w++)
                x_full[h * Win + w] = hist_out[hist_start + h * Win + w];
        for (int h = 0; h < H_in; h++)
            for (int w = 0; w < Win; w++)
                x_full[(*hist_len + h) * Win + w] = x[(chan * H_in + h) * Win + w];

        int H_padded = H_full + 2 * pad_h;
        int W_padded = Win + 2 * pad_w;

        for (int w = 0; w < Wout; w++) y[chan * Wout + w] = bias[chan];

        int16_t *kernel_dila = (int16_t *)calloc(Hk_dila * Wk, sizeof(int16_t));
        for (int hk = 0; hk < Hk; hk++) {
            int hk_dila = (hk == 0) ? 0 : hk * dilation;
            for (int wk = 0; wk < Wk; wk++)
                kernel_dila[hk_dila * Wk + wk] = weight[(chan * Hk + hk) * Wk + wk];
        }

        for (int ho = 0; ho < Hout; ho++) {
            for (int wo = 0; wo < Wout; wo++) {
                int64_t acc = 0;
                for (int hk = 0; hk < Hk_dila; hk++) {
                    for (int wk = 0; wk < Wk; wk++) {
                        int hi = ho + hk - pad_h;
                        int wi = wo + wk - pad_w;
                        int32_t xv = 0;
                        if (hi >= 0 && hi < H_full && wi >= 0 && wi < Win)
                            xv = x_full[hi * Win + wi];
                        int16_t kv = kernel_dila[hk * Wk + wk];
                        if (kv == 0) continue;
                        int64_t prod = (int64_t)xv * (int64_t)kv;
                        int shift = -qr;
                        prod = (prod + ((int64_t)1 << (shift - 1))) >> shift;
                        acc += prod;
                    }
                }
                y[chan * Wout + wo] = sat32((int64_t)y[chan * Wout + wo] + acc);
            }
        }
        free(x_full);
        free(kernel_dila);
    }
}

/* ================================================================
 * ddtconv2d_fixed — Depth-wise Dilated Transposed Conv2D
 * ================================================================
 * weight: (Cout, 1, Hk, Wk) s16f12, kernel pre-rotation rot90(kernel_dila, 2)
 */

void ddtconv2d_fixed(const int32_t *x, int C, int H_in, int Win,
                     const int16_t *weight, const int32_t *bias,
                     int Wout, int Hk, int Wk,
                     int pad_w, int dilation, int qr,
                     int32_t *y,
                     int32_t *hist_out, int *hist_len) {
    int Cout = C;
    int Hout = 1;
    int Hk_dila = Hk + 2 * (dilation - 1);

    for (int chan = 0; chan < Cout; chan++) {
        int H_full = *hist_len + H_in;
        int32_t *x_full = (int32_t *)calloc(H_full * Win, sizeof(int32_t));
        int hist_start = chan * (*hist_len) * Win;
        for (int h = 0; h < *hist_len; h++)
            for (int w = 0; w < Win; w++)
                x_full[h * Win + w] = hist_out[hist_start + h * Win + w];
        for (int h = 0; h < H_in; h++)
            for (int w = 0; w < Win; w++)
                x_full[(*hist_len + h) * Win + w] = x[(chan * H_in + h) * Win + w];

        int H_full_insert = H_full;
        int32_t *x_insert = (int32_t *)calloc(H_full_insert * Win, sizeof(int32_t));
        for (int h = 0; h < H_full; h++)
            for (int w = 0; w < Win; w++)
                x_insert[h * Win + w] = x_full[h * Win + w];

        int pad_w_use = 1;
        int W_padded = Win + 2 * pad_w_use;

        int16_t *kernel_dila = (int16_t *)calloc(Hk_dila * Wk, sizeof(int16_t));
        for (int hk = 0; hk < Hk; hk++) {
            int hk_dila = (hk == 0) ? 0 : hk * dilation;
            for (int wk = 0; wk < Wk; wk++)
                kernel_dila[hk_dila * Wk + wk] = weight[(chan * Hk + hk) * Wk + wk];
        }

        int16_t *kernel_rot = (int16_t *)calloc(Hk_dila * Wk, sizeof(int16_t));
        for (int hk = 0; hk < Hk_dila; hk++)
            for (int wk = 0; wk < Wk; wk++)
                kernel_rot[(Hk_dila - 1 - hk) * Wk + (Wk - 1 - wk)] = kernel_dila[hk * Wk + wk];

        for (int w = 0; w < Wout; w++) y[chan * Wout + w] = bias[chan];

        for (int ho = 0; ho < Hout; ho++) {
            for (int wo = 0; wo < Wout; wo++) {
                int64_t acc = 0;
                for (int hk = 0; hk < Hk_dila; hk++) {
                    for (int wk = 0; wk < Wk; wk++) {
                        int hi = ho + hk;
                        int wi = wo + wk - pad_w_use;
                        int32_t xv = 0;
                        if (hi >= 0 && hi < H_full_insert && wi >= 0 && wi < Win)
                            xv = x_insert[hi * Win + wi];
                        int16_t kv = kernel_rot[hk * Wk + wk];
                        if (kv == 0) continue;
                        int64_t prod = (int64_t)xv * (int64_t)kv;
                        int shift = -qr;
                        prod = (prod + ((int64_t)1 << (shift - 1))) >> shift;
                        acc += prod;
                    }
                }
                y[chan * Wout + wo] = sat32((int64_t)y[chan * Wout + wo] + acc);
            }
        }
        free(x_full); free(x_insert);
        free(kernel_dila); free(kernel_rot);
    }
}

/* ================================================================
 * pconv2d_fixed — Point-wise Conv2D (1×1)
 * ================================================================
 * weight: (Cout, Cin, 1, 1)
 */

void pconv2d_fixed(const int32_t *x, int Cin, int Win,
                   const int16_t *weight, const int32_t *bias,
                   int Cout, int qr,
                   int32_t *y) {
    for (int co = 0; co < Cout; co++) {
        for (int w = 0; w < Win; w++) {
            int64_t acc = bias[co];
            for (int ci = 0; ci < Cin; ci++) {
                int32_t xv = x[ci * Win + w];
                int16_t kv = weight[co * Cin + ci];
                int64_t prod = (int64_t)xv * (int64_t)kv;
                int shift = -qr;
                prod = (prod + ((int64_t)1 << (shift - 1))) >> shift;
                acc += prod;
            }
            y[co * Win + w] = sat32(acc);
        }
    }
}

/* ================================================================
 * ptconv2d_fixed — Point-wise Transposed Conv2D (1×1)
 * ================================================================
 * weight: (Cin, Cout) — (in, out) ordering
 */

void ptconv2d_fixed(const int32_t *x, int Cin, int Win,
                    const int16_t *weight, const int32_t *bias,
                    int Cout, int qr,
                    int32_t *y) {
    for (int co = 0; co < Cout; co++) {
        for (int w = 0; w < Win; w++) {
            int64_t acc = bias[co];
            for (int ci = 0; ci < Cin; ci++) {
                int32_t xv = x[ci * Win + w];
                int16_t kv = weight[ci * Cout + co];
                int64_t prod = (int64_t)xv * (int64_t)kv;
                int shift = -qr;
                prod = (prod + ((int64_t)1 << (shift - 1))) >> shift;
                acc += prod;
            }
            y[co * Win + w] = sat32(acc);
        }
    }
}

/* ================================================================
 * bn_fixed — Batch Normalization
 * ================================================================
 * Matches bn_func.m:
 *   x_norm = round((x - running_mean) .* running_var * 2^qr1)
 *   y = round(x_norm .* weight * 2^qr2) + bias
 */

void bn_fixed(int32_t *x, int C, int Win,
              const uint16_t *weight, const int32_t *bias,
              const int32_t *running_mean, const uint16_t *running_var,
              int qr1, int qr2) {
    for (int c = 0; c < C; c++) {
        int32_t *ch = x + c * Win;
        for (int w = 0; w < Win; w++) {
            int64_t diff = (int64_t)ch[w] - (int64_t)running_mean[c];
            int64_t norm = diff * (int64_t)running_var[c];
            int shift1 = -qr1;
            int32_t x_norm = (int32_t)((norm + ((int64_t)1 << (shift1 - 1))) >> shift1);
            int64_t scaled = (int64_t)x_norm * (int64_t)weight[c];
            int shift2 = -qr2;
            int32_t y_val = (int32_t)((scaled + ((int64_t)1 << (shift2 - 1))) >> shift2);
            ch[w] = sat32((int64_t)y_val + (int64_t)bias[c]);
        }
    }
}

/* ================================================================
 * prelu_fixed — Parametric ReLU
 * ================================================================
 * y(x < 0) = round(x * weight[c] * 2^qr)
 * weight can be scalar (1 element) or per-channel (C elements)
 */

void prelu_fixed(int32_t *x, int C, int Win,
                 const int16_t *weight, int qr) {
    /* All PReLU weights are scalar (shape 1×1), shared across channels */
    int16_t slope = weight[0];
    for (int c = 0; c < C; c++) {
        int32_t *ch = x + c * Win;
        for (int w = 0; w < Win; w++) {
            if (ch[w] < 0) {
                int64_t prod = (int64_t)ch[w] * (int64_t)slope;
                int shift = -qr;
                ch[w] = (int32_t)((prod + ((int64_t)1 << (shift - 1))) >> shift);
            }
        }
    }
}

/* ================================================================
 * ln_fixed — Layer Normalization
 * ================================================================
 * Matches ln_func.m.
 * weight/bias: per-element (same shape as x) or per-channel (C elements)
 */

void ln_fixed(int32_t *x, int C, int Win,
              const int16_t *weight, const int32_t *bias, int qr) {
    /* weight/bias: per-element, shape (Win, C) stored row-major.
     * x: shape (C, Win) row-major. weight index = w*C + c.
     *
     * Fixed-point mean/var: int64 accumulation + isqrt64 + int reciprocal.
     * Eliminates all soft-float ops: 2×float loops, sqrtf, float div. */
    int N = C * Win;

    /* Mean: int64 sum → rounded division */
    int64_t sum_x = 0;
    for (int i = 0; i < N; i++) sum_x += x[i];
    int32_t mean_q = (int32_t)((sum_x + (N >> 1)) / N);  /* rounded Q20 mean */

    /* Variance: Σ(x[i] - mean)^2 in Q40 */
    int64_t sum_sq = 0;
    for (int i = 0; i < N; i++) {
        int64_t diff = (int64_t)x[i] - (int64_t)mean_q;  /* Q20 */
        sum_sq += diff * diff;                             /* Q40 */
    }
    /* var_q40 = sum_sq/N + ε  (ε ≈ 1e-8 × 2^40 = 10995) */
    uint64_t var_q40 = (uint64_t)((sum_sq + (N >> 1)) / N) + 10995ULL;

    /* rms = sqrt(var) in Q20, then reciprocal to Q18: inv = 2^38 / rms */
    uint32_t rms_q20 = isqrt64(var_q40);
    uint32_t var_q = (rms_q20 > 0) ? (uint32_t)((274877906944ULL + (rms_q20 >> 1)) / rms_q20)
                                   : (uint32_t)0xFFFFFFFFU;  /* 2^38 = 274877906944 */

    for (int i = 0; i < N; i++) {
        int64_t diff_q = (int64_t)x[i] - (int64_t)mean_q;
        int64_t norm = diff_q * (int64_t)var_q;
        int32_t x_norm = (int32_t)((norm + ((int64_t)1 << 17)) >> 18);
        int c = i / Win;        /* channel index within x (C-major) */
        int w = i % Win;        /* freq index within x */
        int widx = w * C + c;   /* weight is (Win, C) row-major */
        int64_t scaled = (int64_t)x_norm * (int64_t)weight[widx];
        int shift = -qr;
        x[i] = sat32((int64_t)((scaled + ((int64_t)1 << (shift - 1))) >> shift) + bias[widx]);
    }
}

/* ================================================================
 * gru_fixed — GRU Module
 * ================================================================
 * x: (seq_len, input_dim) s32f20 → y: (seq_len, hidden_dim) s16f15
 * ih_weight: (input_dim, 3*hidden_dim) s16f12
 * Shifts: >>22 (ih), >>17 (hh), sigmoid/tanh Q10→Q15, hidden update >>15
 */

void gru_fixed(const int32_t *x, int seq_len, int input_dim, int hidden_dim,
               const int16_t *ih_weight, const int16_t *ih_bias,
               const int16_t *hh_weight, const int16_t *hh_bias,
               int16_t *y, int16_t *h_prev) {
    int s_ih = 3 * hidden_dim; /* stride: ih_weight shape (input_dim, 3*hidden_dim) */
    int s_hh = 3 * hidden_dim; /* stride: hh_weight shape (hidden_dim, 3*hidden_dim) */
    const int16_t *ih_r_w = ih_weight;
    const int16_t *ih_z_w = ih_weight + hidden_dim;
    const int16_t *ih_n_w = ih_weight + 2 * hidden_dim;
    const int16_t *ih_r_b = ih_bias, *ih_z_b = ih_bias + hidden_dim, *ih_n_b = ih_bias + 2*hidden_dim;
    const int16_t *hh_r_w = hh_weight;
    const int16_t *hh_z_w = hh_weight + hidden_dim;
    const int16_t *hh_n_w = hh_weight + 2 * hidden_dim;
    const int16_t *hh_r_b = hh_bias, *hh_z_b = hh_bias + hidden_dim, *hh_n_b = hh_bias + 2*hidden_dim;

    for (int f = 0; f < seq_len; f++) {
        const int32_t *x_t = x + f * input_dim;
        int16_t *hp = h_prev;

        int32_t r_t[8];
        for (int h = 0; h < hidden_dim; h++) {
            int64_t acc = 0;
            for (int i = 0; i < input_dim; i++) acc += (int64_t)x_t[i] * ih_r_w[i*s_ih+h];
            acc = (acc + ((int64_t)1 << 21)) >> 22;
            int64_t acc_h = 0;
            for (int i = 0; i < hidden_dim; i++) acc_h += (int64_t)hp[i] * hh_r_w[i*s_hh+h];
            acc_h = (acc_h + ((int64_t)1 << 16)) >> 17;
            r_t[h] = sat32(acc + acc_h + ih_r_b[h] + hh_r_b[h]);
        }
        int16_t r_q15[8];
        for (int h = 0; h < hidden_dim; h++) r_q15[h] = Q10_SIGMOID(r_t[h]);

        int32_t z_t[8];
        for (int h = 0; h < hidden_dim; h++) {
            int64_t acc = 0;
            for (int i = 0; i < input_dim; i++) acc += (int64_t)x_t[i] * ih_z_w[i*s_ih+h];
            acc = (acc + ((int64_t)1 << 21)) >> 22;
            int64_t acc_h = 0;
            for (int i = 0; i < hidden_dim; i++) acc_h += (int64_t)hp[i] * hh_z_w[i*s_hh+h];
            acc_h = (acc_h + ((int64_t)1 << 16)) >> 17;
            z_t[h] = sat32(acc + acc_h + ih_z_b[h] + hh_z_b[h]);
        }
        int16_t z_q15[8];
        for (int h = 0; h < hidden_dim; h++) z_q15[h] = Q10_SIGMOID(z_t[h]);

        int32_t h_t[8];
        for (int h = 0; h < hidden_dim; h++) {
            int64_t acc = 0;
            for (int i = 0; i < hidden_dim; i++) acc += (int64_t)hp[i] * hh_n_w[i*s_hh+h];
            h_t[h] = sat32((acc + ((int64_t)1 << 16)) >> 17) + hh_n_b[h];
        }
        int32_t n_t[8];
        for (int h = 0; h < hidden_dim; h++) {
            int64_t acc = 0;
            for (int i = 0; i < input_dim; i++) acc += (int64_t)x_t[i] * ih_n_w[i*s_ih+h];
            acc = (acc + ((int64_t)1 << 21)) >> 22;
            int32_t rh = (int32_t)(((int64_t)r_q15[h] * h_t[h] + 16384) >> 15);
            n_t[h] = sat32(acc + rh + ih_n_b[h]);
        }
        int16_t n_q15[8];
        for (int h = 0; h < hidden_dim; h++) n_q15[h] = Q10_TANH(n_t[h]);

        for (int h = 0; h < hidden_dim; h++) {
            int32_t t1 = (int32_t)(((int64_t)(32768 - z_q15[h]) * n_q15[h] + 16384) >> 15);
            int32_t t2 = (int32_t)(((int64_t)z_q15[h] * hp[h] + 16384) >> 15);
            int16_t hn = sat16(t1 + t2);
            y[f * hidden_dim + h] = hn;
            h_prev[h] = hn;
        }
    }
}

/* ================================================================
 * gru_fixed_perframe — Per-Frame Independent GRU
 * ================================================================
 * Like gru_fixed but each frame has its OWN hidden state,
 * stored at h_prev[f*hidden_dim + h]. Matches Python GRU_module.
 * Used by Inter_RNN_module.
 */
void gru_fixed_perframe(const int32_t *x, int seq_len, int input_dim, int hidden_dim,
                        const int16_t *ih_weight, const int16_t *ih_bias,
                        const int16_t *hh_weight, const int16_t *hh_bias,
                        int16_t *y, int16_t *h_prev,
                        int state_stride) {
    int s_ih = 3 * hidden_dim; /* stride: ih_weight shape (input_dim, 3*hidden_dim) */
    int s_hh = 3 * hidden_dim; /* stride: hh_weight shape (hidden_dim, 3*hidden_dim) */
    const int16_t *ih_r_w = ih_weight;
    const int16_t *ih_z_w = ih_weight + hidden_dim;
    const int16_t *ih_n_w = ih_weight + 2 * hidden_dim;
    const int16_t *ih_r_b = ih_bias, *ih_z_b = ih_bias + hidden_dim, *ih_n_b = ih_bias + 2*hidden_dim;
    const int16_t *hh_r_w = hh_weight;
    const int16_t *hh_z_w = hh_weight + hidden_dim;
    const int16_t *hh_n_w = hh_weight + 2 * hidden_dim;
    const int16_t *hh_r_b = hh_bias, *hh_z_b = hh_bias + hidden_dim, *hh_n_b = hh_bias + 2*hidden_dim;

    for (int f = 0; f < seq_len; f++) {
        const int32_t *x_t = x + f * input_dim;
        int16_t *hp = h_prev + f * state_stride;  /* each frame has independent state, stride=full_dim */

        int32_t r_t[8];
        for (int h = 0; h < hidden_dim; h++) {
            int64_t acc = 0;
            for (int i = 0; i < input_dim; i++) acc += (int64_t)x_t[i] * ih_r_w[i*s_ih+h];
            acc = (acc + ((int64_t)1 << 21)) >> 22;
            int64_t acc_h = 0;
            for (int i = 0; i < hidden_dim; i++) acc_h += (int64_t)hp[i] * hh_r_w[i*s_hh+h];
            acc_h = (acc_h + ((int64_t)1 << 16)) >> 17;
            r_t[h] = sat32(acc + acc_h + ih_r_b[h] + hh_r_b[h]);
        }
        int16_t r_q15[8];
        for (int h = 0; h < hidden_dim; h++) r_q15[h] = Q10_SIGMOID(r_t[h]);

        int32_t z_t[8];
        for (int h = 0; h < hidden_dim; h++) {
            int64_t acc = 0;
            for (int i = 0; i < input_dim; i++) acc += (int64_t)x_t[i] * ih_z_w[i*s_ih+h];
            acc = (acc + ((int64_t)1 << 21)) >> 22;
            int64_t acc_h = 0;
            for (int i = 0; i < hidden_dim; i++) acc_h += (int64_t)hp[i] * hh_z_w[i*s_hh+h];
            acc_h = (acc_h + ((int64_t)1 << 16)) >> 17;
            z_t[h] = sat32(acc + acc_h + ih_z_b[h] + hh_z_b[h]);
        }
        int16_t z_q15[8];
        for (int h = 0; h < hidden_dim; h++) z_q15[h] = Q10_SIGMOID(z_t[h]);

        int32_t h_t[8];
        for (int h = 0; h < hidden_dim; h++) {
            int64_t acc = 0;
            for (int i = 0; i < hidden_dim; i++) acc += (int64_t)hp[i] * hh_n_w[i*s_hh+h];
            h_t[h] = sat32((acc + ((int64_t)1 << 16)) >> 17) + hh_n_b[h];
        }
        int32_t n_t[8];
        for (int h = 0; h < hidden_dim; h++) {
            int64_t acc = 0;
            for (int i = 0; i < input_dim; i++) acc += (int64_t)x_t[i] * ih_n_w[i*s_ih+h];
            acc = (acc + ((int64_t)1 << 21)) >> 22;
            int32_t rh = (int32_t)(((int64_t)r_q15[h] * h_t[h] + 16384) >> 15);
            n_t[h] = sat32(acc + rh + ih_n_b[h]);
        }
        int16_t n_q15[8];
        for (int h = 0; h < hidden_dim; h++) n_q15[h] = Q10_TANH(n_t[h]);

        for (int h = 0; h < hidden_dim; h++) {
            int32_t t1 = (int32_t)(((int64_t)(32768 - z_q15[h]) * n_q15[h] + 16384) >> 15);
            int32_t t2 = (int32_t)(((int64_t)z_q15[h] * hp[h] + 16384) >> 15);
            int16_t hn = sat16(t1 + t2);
            y[f * hidden_dim + h] = hn;
            hp[h] = hn;
        }
    }
}

/* ================================================================
 * bigru_fixed — Bidirectional GRU
 * ================================================================ */

void bigru_fixed(const int32_t *x, int seq_len, int input_dim, int hidden_dim,
                 const int16_t *ih_w, const int16_t *ih_b,
                 const int16_t *hh_w, const int16_t *hh_b,
                 const int16_t *re_ih_w, const int16_t *re_ih_b,
                 const int16_t *re_hh_w, const int16_t *re_hh_b,
                 int16_t *y) {
    int16_t h_fwd[4] = {0};
    int16_t *y_fwd = (int16_t *)calloc(seq_len * hidden_dim, sizeof(int16_t));
    gru_fixed(x, seq_len, input_dim, hidden_dim, ih_w, ih_b, hh_w, hh_b, y_fwd, h_fwd);

    int32_t *x_rev = (int32_t *)calloc(seq_len * input_dim, sizeof(int32_t));
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < input_dim; j++)
            x_rev[i*input_dim+j] = x[(seq_len-1-i)*input_dim+j];

    int16_t h_rev[4] = {0};
    int16_t *y_rev = (int16_t *)calloc(seq_len * hidden_dim, sizeof(int16_t));
    gru_fixed(x_rev, seq_len, input_dim, hidden_dim, re_ih_w, re_ih_b, re_hh_w, re_hh_b, y_rev, h_rev);

    for (int f = 0; f < seq_len; f++) {
        for (int h = 0; h < hidden_dim; h++) {
            y[f*(2*hidden_dim)+h] = y_fwd[f*hidden_dim+h];
            y[f*(2*hidden_dim)+hidden_dim+h] = y_rev[(seq_len-1-f)*hidden_dim+h];
        }
    }
    free(y_fwd); free(x_rev); free(y_rev);
}

/* ================================================================
 * SFE_fixed — Subband Feature Extraction
 * ================================================================ */

void SFE_fixed(const int32_t *x, int C, int Win, int32_t *y) {
    for (int ci = 0; ci < C; ci++) {
        for (int k = 0; k < 3; k++) {
            for (int w = 0; w < Win; w++) {
                int pad_idx = k + w;
                int32_t val = 0;
                if (pad_idx >= 1 && pad_idx <= Win) val = x[ci*Win+(pad_idx-1)];
                y[(ci*3+k)*Win+w] = val;
            }
        }
    }
}

/* ================================================================
 * BM_fixed — Band Merging: (3, 257) → (3, 129)
 * Matches Python BM_module: y[:,:65]=x[:,:65], y[:,65:]=round(x[:,65:] @ W / 2^15)
 * weight shape: (192, 64), Q-format: u16f15
 * x[:,65:] shape: (3, 192), weight shape: (192, 64), output: (3, 64)
 * ================================================================ */

void BM_fixed(const int32_t *x, const uint16_t *weight, int32_t *y) {
    int Win = 257, Wout = 129;
    int M = Win - 65;  /* 192 input elements in the projection region */
    int N = Wout - 65; /* 64 output elements */

    for (int c = 0; c < 3; c++) {
        /* Low-frequency passthrough: w=0..64 */
        for (int w = 0; w < 65; w++) y[c*Wout+w] = x[c*Win+w];

        /* High-frequency projection: full matmul x[c,65:] @ weight / 2^15 */
        for (int j = 0; j < N; j++) {
            int64_t acc = 0;
            for (int i = 0; i < M; i++) {
                acc += (int64_t)x[c*Win + 65 + i] * weight[i * N + j];
            }
            y[c*Wout + 65 + j] = (int32_t)((acc + 16384) >> 15);
        }
    }
}

/* ================================================================
 * BS_fixed — Band Splitting: (2, 129) → (2, 257)
 * ================================================================ */

void BS_fixed(const int16_t *x, const uint16_t *weight, int16_t *y) {
    int Win = 129, Wout = 257;
    int M = Win - 65;   /* 64 input elements in the projection region */
    int N = Wout - 65;  /* 192 output elements */

    for (int c = 0; c < 2; c++) {
        /* Low-frequency passthrough: w=0..64 */
        for (int w = 0; w < 65; w++) y[c*Wout+w] = x[c*Win+w];

        /* High-frequency projection: full matmul x[c,65:] @ weight / 2^15 */
        for (int j = 0; j < N; j++) {
            int64_t acc = 0;
            for (int i = 0; i < M; i++) {
                acc += (int64_t)x[c*Win + 65 + i] * weight[i * N + j];
            }
            y[c*Wout + 65 + j] = (int16_t)((acc + 16384) >> 15);
        }
    }
}

/* ================================================================
 * MASK_fixed — Complex Ratio Mask
 * ================================================================ */

void MASK_fixed(const int16_t *mask, const int32_t *real_in,
                const int32_t *imag_in, int32_t *y) {
    for (int w = 0; w < 257; w++) {
        int64_t yr = (int64_t)real_in[w]*mask[w] - (int64_t)imag_in[w]*mask[257+w];
        y[w] = sat32((yr + 16384) >> 15);
        int64_t yi = (int64_t)imag_in[w]*mask[w] + (int64_t)real_in[w]*mask[257+w];
        y[257+w] = sat32((yi + 16384) >> 15);
    }
}

/* ================================================================
 * mag_gen — Input magnitude generation
 * ================================================================ */

void mag_gen(const float *real_in, const float *imag_in, int N, int32_t *y) {
    for (int i = 0; i < N; i++) {
        int32_t r_q20 = F2Q20(real_in[i]);
        int32_t i_q20 = F2Q20(imag_in[i]);
        y[N+i]   = r_q20;
        y[2*N+i] = i_q20;
        uint64_t sum_sq = (uint64_t)r_q20 * (uint64_t)r_q20
                        + (uint64_t)i_q20 * (uint64_t)i_q20;
        y[i] = (int32_t)isqrt64(sum_sq);
    }
}

/* Q15 integer version — skips float round-trip.
 * real_in, imag_in in Q15 (int32_t). Q15<<5 = Q20, exact (no round). */
void mag_gen_q15(const int32_t *real_in, const int32_t *imag_in, int N, int32_t *y) {
    for (int i = 0; i < N; i++) {
        int32_t r_q20 = (int32_t)((int64_t)real_in[i] << 5);
        int32_t i_q20 = (int32_t)((int64_t)imag_in[i] << 5);
        y[N+i]   = r_q20;
        y[2*N+i] = i_q20;
        uint64_t sum_sq = (uint64_t)r_q20 * (uint64_t)r_q20
                        + (uint64_t)i_q20 * (uint64_t)i_q20;
        y[i] = (int32_t)isqrt64(sum_sq);
    }
}

/* ================================================================
 * State Initialization
 * ================================================================ */

void denoise_state_init(denoise_state_t *s) {
    memset(s, 0, sizeof(denoise_state_t));
}

/* ================================================================
 * TRA_module — Temporal Recurrent Attention
 * ================================================================
 * x: (8, 33) s32f20. Agg: square+mean → GRU(8→16) + FC(16→8) + Sigmoid
 * Single hidden state h_prev[16] s16f15 — ONE GRU step, ONE attention
 * vector broadcast across all 33 frequency bins. Matches TRA_module.m.
 * Shifts: ih>>13, hh>>8, fc>>8, att>>15
 */

void TRA_module(const int32_t *x, int16_t *h_prev, int gtc_idx,
                const int16_t *ih_w, const int32_t *ih_b,
                const int16_t *hh_w, const int32_t *hh_b,
                const int16_t *fc_w, const int32_t *fc_b,
                int32_t *y) {
    int Cin = 8, Win = 33, hidden = 16;
    (void)gtc_idx;

    /* === Aggregation: square + mean → (8,) u32f20 (int64, no float) === */
    int32_t x_agg[8];
    for (int c = 0; c < Cin; c++) {
        int64_t sum_sq = 0;
        for (int w = 0; w < Win; w++) {
            int64_t v = x[c*Win+w];  /* s32f20 */
            sum_sq += v * v;          /* Q40 */
        }
        /* Divide by Win=33, Q40→Q20: +2^19 for rounding, >>20 */
        x_agg[c] = (int32_t)((sum_sq / Win + (1 << 19)) >> 20);
    }

    int stride_ih = 3 * hidden;
    int stride_hh = 3 * hidden;
    const int16_t *ih_r_w = ih_w, *ih_z_w = ih_w + hidden, *ih_n_w = ih_w + 2*hidden;
    const int32_t *ih_r_b = ih_b, *ih_z_b = ih_b + hidden, *ih_n_b = ih_b + 2*hidden;
    const int16_t *hh_r_w = hh_w, *hh_z_w = hh_w + hidden, *hh_n_w = hh_w + 2*hidden;
    const int32_t *hh_r_b = hh_b, *hh_z_b = hh_b + hidden, *hh_n_b = hh_b + 2*hidden;

    int16_t *hp = h_prev;  /* single state (16,) — NOT per-bin! */

    /* === GRU: ONE step on x_agg (8→16) === */

    /* R gate */
    int32_t r_t[16];
    for (int h = 0; h < hidden; h++) {
        int64_t acc = 0;
        for (int i = 0; i < Cin; i++) acc += (int64_t)x_agg[i] * ih_r_w[i*stride_ih+h];
        acc = (acc + 4096) >> 13;
        int64_t acc_h = 0;
        for (int i = 0; i < hidden; i++) acc_h += (int64_t)hp[i] * hh_r_w[i*stride_hh+h];
        acc_h = (acc_h + 128) >> 8;
        r_t[h] = sat32(acc + acc_h + ih_r_b[h] + hh_r_b[h]);
    }
    int16_t r_q15[16];
    for (int h = 0; h < hidden; h++) r_q15[h] = Q20_SIGMOID(r_t[h]);

    /* Z gate */
    int32_t z_t[16];
    for (int h = 0; h < hidden; h++) {
        int64_t acc = 0;
        for (int i = 0; i < Cin; i++) acc += (int64_t)x_agg[i] * ih_z_w[i*stride_ih+h];
        acc = (acc + 4096) >> 13;
        int64_t acc_h = 0;
        for (int i = 0; i < hidden; i++) acc_h += (int64_t)hp[i] * hh_z_w[i*stride_hh+h];
        acc_h = (acc_h + 128) >> 8;
        z_t[h] = sat32(acc + acc_h + ih_z_b[h] + hh_z_b[h]);
    }
    int16_t z_q15[16];
    for (int h = 0; h < hidden; h++) z_q15[h] = Q20_SIGMOID(z_t[h]);

    /* N gate */
    int32_t h_t[16];
    for (int h = 0; h < hidden; h++) {
        int64_t acc = 0;
        for (int i = 0; i < hidden; i++) acc += (int64_t)hp[i] * hh_n_w[i*stride_hh+h];
        h_t[h] = sat32((acc + 128) >> 8) + hh_n_b[h];
    }
    int32_t n_t[16];
    for (int h = 0; h < hidden; h++) {
        int64_t acc = 0;
        for (int i = 0; i < Cin; i++) acc += (int64_t)x_agg[i] * ih_n_w[i*stride_ih+h];
        acc = (acc + 4096) >> 13;
        int32_t rh = (int32_t)(((int64_t)r_q15[h]*h_t[h] + 16384) >> 15);
        n_t[h] = sat32(acc + rh + ih_n_b[h]);
    }
    int16_t n_q15[16];
    for (int h = 0; h < hidden; h++) n_q15[h] = Q20_TANH(n_t[h]);

    /* Hidden state update (single state!) */
    for (int h = 0; h < hidden; h++) {
        int32_t t1 = (int32_t)(((int64_t)(32768-z_q15[h])*n_q15[h] + 16384) >> 15);
        int32_t t2 = (int32_t)(((int64_t)z_q15[h]*hp[h] + 16384) >> 15);
        hp[h] = sat16(t1 + t2);
    }

    /* === FC: (16→8) + Sigmoid → attention (8,) === */
    /* MATLAB: x_fc = round(x_gru * fc_weight * 2^(-8)) + fc_bias */
    /* Bias (s32f20) added AFTER shift — NOT inside accumulator! */
    int16_t att[8];
    for (int co = 0; co < Cin; co++) {
        int64_t acc = 0;
        for (int h = 0; h < hidden; h++) acc += (int64_t)hp[h] * fc_w[h*Cin+co];
        int32_t fc_out = sat32((int64_t)((acc + 128) >> 8) + fc_b[co]);
        att[co] = Q20_SIGMOID(fc_out);
    }

    /* === Apply attention: broadcast across all 33 frequency bins === */
    for (int co = 0; co < Cin; co++) {
        for (int f = 0; f < Win; f++) {
            y[co*Win+f] = (int32_t)(((int64_t)x[co*Win+f]*att[co] + 16384) >> 15);
        }
    }
}

/* ================================================================
 * DeTRA_module — Decoder Temporal Recurrent Attention
 * ================================================================
 * Differs from TRA: x_agg u32f18, bias s32f18, hh>>10 (not >>8),
 * dequant Q18_TO_F (not Q20_TO_F). Single hidden state, ONE GRU step.
 */

void DeTRA_module(const int32_t *x, int16_t *h_prev, int gtc_idx,
                  const int16_t *ih_w, const int32_t *ih_b,
                  const int16_t *hh_w, const int32_t *hh_b,
                  const int16_t *fc_w, const int32_t *fc_b,
                  int32_t *y) {
    int Cin = 8, Win = 33, hidden = 16;
    (void)gtc_idx;

    /* === Aggregation: square + mean → (8,) u32f18 (int64, no float) === */
    int32_t x_agg[8];
    for (int c = 0; c < Cin; c++) {
        int64_t sum_sq = 0;
        for (int w = 0; w < Win; w++) {
            int64_t v = x[c*Win+w];  /* s32f20 */
            sum_sq += v * v;          /* Q40 */
        }
        /* Divide by Win=33, Q40→Q18: +2^21 for rounding, >>22 */
        x_agg[c] = (int32_t)((sum_sq / Win + (1 << 21)) >> 22);
    }

    int stride_ih = 3 * hidden;
    int stride_hh = 3 * hidden;
    const int16_t *ih_r_w = ih_w, *ih_z_w = ih_w + hidden, *ih_n_w = ih_w + 2*hidden;
    const int32_t *ih_r_b = ih_b, *ih_z_b = ih_b + hidden, *ih_n_b = ih_b + 2*hidden;
    const int16_t *hh_r_w = hh_w, *hh_z_w = hh_w + hidden, *hh_n_w = hh_w + 2*hidden;
    const int32_t *hh_r_b = hh_b, *hh_z_b = hh_b + hidden, *hh_n_b = hh_b + 2*hidden;

    int16_t *hp = h_prev;  /* single state (16,) */

    /* === GRU: ONE step on x_agg === */

    /* R gate — bias s32f18, dequant Q18→float */
    int32_t r_t[16];
    for (int h = 0; h < hidden; h++) {
        int64_t acc = 0;
        for (int i = 0; i < Cin; i++) acc += (int64_t)x_agg[i] * ih_r_w[i*stride_ih+h];
        acc = (acc + 4096) >> 13;
        int64_t acc_h = 0;
        for (int i = 0; i < hidden; i++) acc_h += (int64_t)hp[i] * hh_r_w[i*stride_hh+h];
        acc_h = (acc_h + 512) >> 10;
        r_t[h] = sat32(acc + acc_h + ih_r_b[h] + hh_r_b[h]);
    }
    int16_t r_q15[16];
    for (int h = 0; h < hidden; h++) r_q15[h] = Q18_SIGMOID(r_t[h]);

    /* Z gate */
    int32_t z_t[16];
    for (int h = 0; h < hidden; h++) {
        int64_t acc = 0;
        for (int i = 0; i < Cin; i++) acc += (int64_t)x_agg[i] * ih_z_w[i*stride_ih+h];
        acc = (acc + 4096) >> 13;
        int64_t acc_h = 0;
        for (int i = 0; i < hidden; i++) acc_h += (int64_t)hp[i] * hh_z_w[i*stride_hh+h];
        acc_h = (acc_h + 512) >> 10;
        z_t[h] = sat32(acc + acc_h + ih_z_b[h] + hh_z_b[h]);
    }
    int16_t z_q15[16];
    for (int h = 0; h < hidden; h++) z_q15[h] = Q18_SIGMOID(z_t[h]);

    /* N gate */
    int32_t h_t[16];
    for (int h = 0; h < hidden; h++) {
        int64_t acc = 0;
        for (int i = 0; i < hidden; i++) acc += (int64_t)hp[i] * hh_n_w[i*stride_hh+h];
        h_t[h] = sat32((acc + 512) >> 10) + hh_n_b[h];
    }
    int32_t n_t[16];
    for (int h = 0; h < hidden; h++) {
        int64_t acc = 0;
        for (int i = 0; i < Cin; i++) acc += (int64_t)x_agg[i] * ih_n_w[i*stride_ih+h];
        acc = (acc + 4096) >> 13;
        int32_t rh = (int32_t)(((int64_t)r_q15[h]*h_t[h] + 16384) >> 15);
        n_t[h] = sat32(acc + rh + ih_n_b[h]);
    }
    int16_t n_q15[16];
    for (int h = 0; h < hidden; h++) n_q15[h] = Q18_TANH(n_t[h]);

    /* Hidden state update */
    for (int h = 0; h < hidden; h++) {
        int32_t t1 = (int32_t)(((int64_t)(32768-z_q15[h])*n_q15[h] + 16384) >> 15);
        int32_t t2 = (int32_t)(((int64_t)z_q15[h]*hp[h] + 16384) >> 15);
        hp[h] = sat16(t1 + t2);
    }

    /* === FC: (16→8) + Sigmoid → attention (8,) === */
    /* MATLAB: x_fc = round(x_gru * fc_weight * 2^(-8)) + fc_bias */
    /* Bias (s32f20) added AFTER shift — NOT inside accumulator! */
    int16_t att[8];
    for (int co = 0; co < Cin; co++) {
        int64_t acc = 0;
        for (int h = 0; h < hidden; h++) acc += (int64_t)hp[h] * fc_w[h*Cin+co];
        int32_t fc_out = sat32((int64_t)((acc + 128) >> 8) + fc_b[co]);
        att[co] = Q20_SIGMOID(fc_out);
    }

    /* === Apply attention: broadcast across all 33 frequency bins === */
    for (int co = 0; co < Cin; co++) {
        for (int f = 0; f < Win; f++) {
            y[co*Win+f] = (int32_t)(((int64_t)x[co*Win+f]*att[co] + 16384) >> 15);
        }
    }
}

/* ================================================================
 * Conv_block_0 — Conv2D(9→16, 129→65) → BN → PReLU
 * ================================================================ */

void Conv_block_0(const int32_t *x,
                  const int32_t *conv_w, const int32_t *conv_b,
                  const uint16_t *bn_w, const int32_t *bn_b,
                  const int32_t *bn_m, const uint16_t *bn_v,
                  const int16_t *prelu_w,
                  int32_t *y) {
    int Cin = 9, Win = 129, Cout = 16, Wout = 65, Hk = 1, Wk = 5, stride = 2, pad_w = 2;
    int32_t *y_conv = (int32_t *)calloc(Cout*Wout, sizeof(int32_t));
    for (int co = 0; co < Cout; co++)
        for (int wo = 0; wo < Wout; wo++) y_conv[co*Wout+wo] = conv_b[co];
    for (int co = 0; co < Cout; co++) {
        for (int ci = 0; ci < Cin; ci++) {
            const int32_t *xc = x + ci*Win;
            for (int wo = 0; wo < Wout; wo++) {
                int64_t acc = 0;
                for (int wk = 0; wk < Wk; wk++) {
                    int wi = wo*stride + wk - pad_w;
                    int32_t xv = (wi>=0 && wi<Win) ? xc[wi] : 0;
                    int32_t kv = conv_w[((co*Cin+ci)*Hk)*Wk+wk];
                    acc += ((int64_t)xv*kv + (1<<17)) >> 18;
                }
                y_conv[co*Wout+wo] = sat32((int64_t)y_conv[co*Wout+wo] + acc);
            }
        }
    }
    bn_fixed(y_conv, Cout, Wout, bn_w, bn_b, bn_m, bn_v, -14, -14);
    prelu_fixed(y_conv, Cout, Wout, prelu_w, -14);
    memcpy(y, y_conv, Cout*Wout*sizeof(int32_t));
    free(y_conv);
}

/* ================================================================
 * Conv_block_1 — Grouped Conv2D(16→16, 65→33) → BN → PReLU
 * ================================================================ */

void Conv_block_1(const int32_t *x,
                  const int16_t *conv_w, const int32_t *conv_b,
                  const uint16_t *bn_w, const int32_t *bn_b,
                  const int32_t *bn_m, const uint16_t *bn_v,
                  const int16_t *prelu_w,
                  int32_t *y) {
    int Cin = 8, Win = 65, Cout = 8, Wout = 33, Hk = 1, Wk = 5, stride = 2, pad_w = 2, qr = -13;
    int32_t buf[16*33];
    conv2d_fixed(x, Cin, Win, conv_w, conv_b, Cout, Wout, Hk, Wk, stride, pad_w, qr, buf);
    conv2d_fixed(x+Cin*Win, Cin, Win, conv_w+8*Cin*Hk*Wk, conv_b+8, Cout, Wout, Hk, Wk, stride, pad_w, qr, buf+8*Wout);
    bn_fixed(buf, 16, Wout, bn_w, bn_b, bn_m, bn_v, -10, -14);
    prelu_fixed(buf, 16, Wout, prelu_w, -14);
    memcpy(y, buf, 16*Wout*sizeof(int32_t));
}

/* ================================================================
 * DeConv_block_0 — Transposed Conv2D(16→2, 65→129) + skip → BN → Tanh
 * ================================================================ */

void DeConv_block_0(const int32_t *x_in, const int32_t *x_skip,
                    const int32_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    int16_t *y) {
    int Cin = 16, Win = 65, Cout = 2, Wout = 129, Hk = 1, Wk = 5, stride = 2, pad_w = 2;
    int W_insert = Win + (Win-1)*(stride-1), W_padded = W_insert + 2*pad_w;

    int32_t *x_sum = (int32_t *)calloc(Cin*Win, sizeof(int32_t));
    for (int i = 0; i < Cin*Win; i++) x_sum[i] = sat32((int64_t)x_in[i]+x_skip[i]);

    int32_t *y_conv = (int32_t *)calloc(Cout*Wout, sizeof(int32_t));
    for (int co = 0; co < Cout; co++)
        for (int wo = 0; wo < Wout; wo++) y_conv[co*Wout+wo] = conv_b[co];

    for (int co = 0; co < Cout; co++) {
        for (int ci = 0; ci < Cin; ci++) {
            const int32_t *xc = x_sum + ci*Win;
            int32_t *xins = (int32_t *)calloc(W_padded, sizeof(int32_t));
            for (int w = 0; w < Win; w++) xins[pad_w+w*stride] = xc[w];
            for (int wo = 0; wo < Wout; wo++) {
                int64_t acc = 0;
                for (int wk = 0; wk < Wk; wk++) {
                    int wi = wo + wk, wk_rev = Wk-1-wk;
                    int32_t xv = (wi>=0 && wi<W_padded) ? xins[wi] : 0;
                    int32_t kv = conv_w[((ci*Cout+co)*Hk)*Wk+wk_rev];
                    acc += ((int64_t)xv*kv + (1<<17)) >> 18;
                }
                y_conv[co*Wout+wo] = sat32((int64_t)y_conv[co*Wout+wo]+acc);
            }
            free(xins);
        }
    }
    bn_fixed(y_conv, Cout, Wout, bn_w, bn_b, bn_m, bn_v, -14, -14);
    for (int i = 0; i < Cout*Wout; i++)
        y[i] = Q20_TANH(y_conv[i]);
    free(x_sum); free(y_conv);
}

/* ================================================================
 * DeConv_block_1 — Grouped Transposed Conv2D(16→16, 33→65) + skip → BN → PReLU
 * ================================================================ */

void DeConv_block_1(const int32_t *x_in, const int32_t *x_skip,
                    const int16_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    const int16_t *prelu_w,
                    int32_t *y) {
    int Cin = 8, Win = 33, Cout = 8, Wout = 65, Hk = 1, Wk = 5, stride = 2, qr = -13;
    int32_t x_sum[16*33], buf[16*65];
    for (int i = 0; i < 16*33; i++) x_sum[i] = sat32((int64_t)x_in[i]+x_skip[i]);
    tconv2d_fixed(x_sum, Cin, Win, conv_w, conv_b, Cout, Wout, Hk, Wk, stride, qr, buf);
    tconv2d_fixed(x_sum+8*Win, Cin, Win, conv_w+8*Cin*Hk*Wk, conv_b+8, Cout, Wout, Hk, Wk, stride, qr, buf+8*Wout);
    bn_fixed(buf, 16, Wout, bn_w, bn_b, bn_m, bn_v, -14, -14);
    prelu_fixed(buf, 16, Wout, prelu_w, -14);
    memcpy(y, buf, 16*Wout*sizeof(int32_t));
}

/* ================================================================
 * P_Conv_block_0 — Point-wise Conv2D(24→16) → BN → PReLU
 * ================================================================ */

void P_Conv_block_0(const int32_t *x, int gtc_idx,
                    const int16_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    const int16_t *prelu_w,
                    int32_t *y) {
    int Cin = 24, Win = 33, Cout = 16, qr = -13;
    (void)gtc_idx;
    pconv2d_fixed(x, Cin, Win, conv_w, conv_b, Cout, qr, y);
    bn_fixed(y, Cout, Win, bn_w, bn_b, bn_m, bn_v, -13, -14);
    prelu_fixed(y, Cout, Win, prelu_w, -14);
}

/* ================================================================
 * P_Conv_block_1 — Point-wise Conv2D(16→8) → BN (no PReLU)
 * ================================================================ */

void P_Conv_block_1(const int32_t *x, int gtc_idx,
                    const int16_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_m, const uint16_t *bn_v,
                    int32_t *y) {
    int Cin = 16, Win = 33, Cout = 8, qr = -13;
    (void)gtc_idx;
    pconv2d_fixed(x, Cin, Win, conv_w, conv_b, Cout, qr, y);
    bn_fixed(y, Cout, Win, bn_w, bn_b, bn_m, bn_v, -14, -14);
}

/* ================================================================
 * P_DeConv_block_0 — Point-wise TConv2D(24→16) → BN → PReLU
 * ================================================================ */

void P_DeConv_block_0(const int32_t *x, int gtc_idx,
                      const int16_t *conv_w, const int32_t *conv_b,
                      const uint16_t *bn_w, const int32_t *bn_b,
                      const int32_t *bn_m, const uint16_t *bn_v,
                      const int16_t *prelu_w,
                      int32_t *y) {
    int Cin = 24, Win = 33, Cout = 16, qr = -13;
    (void)gtc_idx;
    ptconv2d_fixed(x, Cin, Win, conv_w, conv_b, Cout, qr, y);
    /* running_var is u16f13 → qr1=-13 (matches encoder P_Conv_block_0) */
    bn_fixed(y, Cout, Win, bn_w, bn_b, bn_m, bn_v, -13, -14);
    prelu_fixed(y, Cout, Win, prelu_w, -14);
}

/* ================================================================
 * P_DeConv_block_1 — Point-wise TConv2D(16→8) → BN (no PReLU)
 * ================================================================ */

void P_DeConv_block_1(const int32_t *x, int gtc_idx,
                      const int16_t *conv_w, const int32_t *conv_b,
                      const uint16_t *bn_w, const int32_t *bn_b,
                      const int32_t *bn_m, const uint16_t *bn_v,
                      int32_t *y) {
    int Cin = 16, Win = 33, Cout = 8, qr = -13;
    (void)gtc_idx;
    ptconv2d_fixed(x, Cin, Win, conv_w, conv_b, Cout, qr, y);
    bn_fixed(y, Cout, Win, bn_w, bn_b, bn_m, bn_v, -14, -14);
}

/* ================================================================
 * DD-Conv with history (inline helper for GT_Conv_module)
 * ================================================================
 * hist_buf: (C=16, hist_len, Win=33) — slice within enc_conv_hist or dec_conv_hist
 * hist_len: number of past frames stored (2/4/10 depending on dilation)
 * x_new: (16, 33) — current frame
 *
 * Operation (matches MATLAB DD_Conv_block.m):
 *   x_shap = reshape(x_new, [16,1,33])
 *   x_inp = cat(2, hist_buf, x_shap)   → (16, hist_len+1, 33)
 *   y_conv = ddconv2d(x_inp, dilation, ...)
 *   y = BN(y_conv) → PReLU(y)
 *   hist_buf = x_inp(:,2:end,:)         → (16, hist_len, 33)
 */

/* Helper: read one frame from state buffer (C, full_time, Win) at time offset t */
static inline const int32_t* hist_frame(const int32_t *state, int c, int t,
                                         int full_time, int Win) {
    return state + c * (full_time * Win) + t * Win;
}
static inline int32_t* hist_frame_mut(int32_t *state, int c, int t,
                                       int full_time, int Win) {
    return state + c * (full_time * Win) + t * Win;
}

/* dd_conv_with_hist: operates on a time-slice within a (C=16, full_time, Win=33) state buffer.
 * time_offset: start index within the full_time dimension of the slice
 * hist_len: number of past frames (2/4/10)
 * The slice occupies time indices [time_offset, time_offset+hist_len).
 */
static void dd_conv_with_hist(int32_t *hist_state, int full_time, int time_offset, int hist_len,
                              const int32_t *x_new,
                              const int16_t *conv_w, const int32_t *conv_b,
                              const uint16_t *bn_w, const int32_t *bn_b,
                              const int32_t *bn_mean, const uint16_t *bn_var,
                              const int16_t *prelu_w,
                              int dilation, int32_t *y) {
    int C = 16, Win = 33;
    int Hk = 3, Wk = 3;
    int pad_h = 0, pad_w = 1;
    int qr_conv = -13, qr_bn1 = -10, qr_bn2 = -14, qr_prelu = -14;

    /* Build x_inp: (C, hist_len+1, Win) = [hist_slice, x_new] */
    int H_full = hist_len + 1;
    int32_t *x_inp = (int32_t *)calloc(C * H_full * Win, sizeof(int32_t));
    /* Copy history from state slice */
    for (int c = 0; c < C; c++)
        for (int h = 0; h < hist_len; h++)
            memcpy(x_inp + (c*H_full+h)*Win,
                   hist_frame(hist_state, c, time_offset+h, full_time, Win),
                   Win*sizeof(int32_t));
    /* Append new frame */
    for (int c = 0; c < C; c++)
        memcpy(x_inp + (c*H_full+hist_len)*Win, x_new + c*Win, Win*sizeof(int32_t));

    /* ddconv2d: depth-wise dilated conv */
    int32_t *y_conv = (int32_t *)calloc(C * Win, sizeof(int32_t));
    {
        int Cout = C;
        int Hout = 1;
        int Hk_dila = Hk + 2*(dilation-1);

        for (int chan = 0; chan < Cout; chan++) {
            for (int w = 0; w < Win; w++) y_conv[chan*Win+w] = conv_b[chan];

            const int32_t *xc = x_inp + chan * H_full * Win;

            /* Build dilated kernel for this channel */
            int16_t kd[33]; /* max Hk_dila*Wk = 11*3 = 33 */
            memset(kd, 0, sizeof(kd));
            for (int hk = 0; hk < Hk; hk++) {
                int hkd = (hk == 0) ? 0 : hk * dilation;
                for (int wk = 0; wk < Wk; wk++)
                    kd[hkd*Wk+wk] = conv_w[(chan*Hk+hk)*Wk+wk];
            }

            for (int wo = 0; wo < Win; wo++) {
                int64_t acc = 0;
                for (int hk = 0; hk < Hk_dila; hk++) {
                    for (int wk = 0; wk < Wk; wk++) {
                        int hi = hk - pad_h;
                        int wi = wo + wk - pad_w;
                        int32_t xv = (hi>=0 && hi<H_full && wi>=0 && wi<Win) ? xc[hi*Win+wi] : 0;
                        int16_t kv = kd[hk*Wk+wk];
                        if (kv == 0) continue;
                        int64_t prod = (int64_t)xv * kv;
                        prod = (prod + (1<<12)) >> 13; /* qr_conv=-13 */
                        acc += prod;
                    }
                }
                y_conv[chan*Win+wo] = sat32((int64_t)y_conv[chan*Win+wo] + acc);
            }
        }
    }

    /* BN */
    bn_fixed(y_conv, C, Win, bn_w, bn_b, bn_mean, bn_var, qr_bn1, qr_bn2);

    /* PReLU */
    prelu_fixed(y_conv, C, Win, prelu_w, qr_prelu);

    /* Copy to output */
    memcpy(y, y_conv, C*Win*sizeof(int32_t));

    /* Update history: drop oldest frame, shift left */
    for (int c = 0; c < C; c++)
        for (int h = 0; h < hist_len; h++)
            memcpy(hist_frame_mut(hist_state, c, time_offset+h, full_time, Win),
                   x_inp + (c*H_full+h+1)*Win, Win*sizeof(int32_t));

    free(x_inp);
    free(y_conv);
}

/* ================================================================
 * DD-DeConv with history (inline helper for GT_DeConv_module)
 * ================================================================
 * Same pattern as dd_conv_with_hist but uses transposed (ddtconv2d)
 * and decoder Q-formats: conv s16f12, BN var u16f12, qr_conv=-12, qr_bn1=-12
 */

void dd_deconv_with_hist(int32_t *hist_state, int full_time, int time_offset, int hist_len,
                                const int32_t *x_new,
                                const int16_t *conv_w, const int32_t *conv_b,
                                const uint16_t *bn_w, const int32_t *bn_b,
                                const int32_t *bn_mean, const uint16_t *bn_var,
                                const int16_t *prelu_w,
                                int dilation, int32_t *y) {
    int C = 16, Win = 33;
    int Hk = 3, Wk = 3;
    int pad_w = 1;
    int qr_conv = -12, qr_bn1 = -12, qr_bn2 = -14, qr_prelu = -14;

    /* Build x_inp: (C, hist_len+1, Win) from state slice + new frame */
    int H_full = hist_len + 1;
    int32_t *x_inp = (int32_t *)calloc(C * H_full * Win, sizeof(int32_t));
    for (int c = 0; c < C; c++)
        for (int h = 0; h < hist_len; h++)
            memcpy(x_inp + (c*H_full+h)*Win,
                   hist_frame(hist_state, c, time_offset+h, full_time, Win),
                   Win*sizeof(int32_t));
    for (int c = 0; c < C; c++)
        memcpy(x_inp + (c*H_full+hist_len)*Win, x_new + c*Win, Win*sizeof(int32_t));

    /* ddtconv2d */
    int32_t *y_conv = (int32_t *)calloc(C * Win, sizeof(int32_t));
    {
        int Cout = C;
        int Hk_dila = Hk + 2*(dilation-1);

        for (int chan = 0; chan < Cout; chan++) {
            for (int w = 0; w < Win; w++) y_conv[chan*Win+w] = conv_b[chan];

            const int32_t *xc = x_inp + chan * H_full * Win;

            /* Build + pre-rotate dilated kernel */
            int16_t kd[33];
            memset(kd, 0, sizeof(kd));
            for (int hk = 0; hk < Hk; hk++) {
                int hkd = (hk == 0) ? 0 : hk * dilation;
                for (int wk = 0; wk < Wk; wk++)
                    kd[hkd*Wk+wk] = conv_w[(chan*Hk+hk)*Wk+wk];
            }
            /* rot90(kernel_dila, 2): reverse both dims */
            int16_t kr[33];
            memset(kr, 0, sizeof(kr));
            for (int hk = 0; hk < Hk_dila; hk++)
                for (int wk = 0; wk < Wk; wk++)
                    kr[(Hk_dila-1-hk)*Wk+(Wk-1-wk)] = kd[hk*Wk+wk];

            for (int wo = 0; wo < Win; wo++) {
                int64_t acc = 0;
                for (int hk = 0; hk < Hk_dila; hk++) {
                    for (int wk = 0; wk < Wk; wk++) {
                        int hi = hk;
                        int wi = wo + wk - pad_w;
                        int32_t xv = (hi>=0 && hi<H_full && wi>=0 && wi<Win) ? xc[hi*Win+wi] : 0;
                        int16_t kv = kr[hk*Wk+wk];
                        if (kv == 0) continue;
                        int64_t prod = (int64_t)xv * kv;
                        prod = (prod + (1<<11)) >> 12; /* qr_conv=-12 */
                        acc += prod;
                    }
                }
                y_conv[chan*Win+wo] = sat32((int64_t)y_conv[chan*Win+wo] + acc);
            }
        }
    }

    /* BN */
    bn_fixed(y_conv, C, Win, bn_w, bn_b, bn_mean, bn_var, qr_bn1, qr_bn2);

    /* PReLU */
    prelu_fixed(y_conv, C, Win, prelu_w, qr_prelu);

    memcpy(y, y_conv, C*Win*sizeof(int32_t));

    /* Update history */
    for (int c = 0; c < C; c++)
        for (int h = 0; h < hist_len; h++)
            memcpy(hist_frame_mut(hist_state, c, time_offset+h, full_time, Win),
                   x_inp + (c*H_full+h+1)*Win, Win*sizeof(int32_t));

    free(x_inp);
    free(y_conv);
}

/* ================================================================
 * GT_Conv_module — Grouped Temporal Convolution Block
 * ================================================================
 * x(16,33) → split x1(8ch)→SFE→PC0→DD-Conv(with history)→PC1→TRA, x2(8ch) pass-thru
 * Channel shuffle: y[0:2:16]=x1_tra, y[1:2:16]=x2
 *
 * conv_hist: base pointer to state->enc_conv_hist (C=16, full_time=16, Win=33)
 * full_time: total time dimension (16)
 * time_offset: start index of this DD block's history slice
 *   dil=1: time_offset=0,  hist_len=2
 *   dil=2: time_offset=2,  hist_len=4
 *   dil=5: time_offset=6,  hist_len=10
 */

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
                    int32_t *y) {
    int Win = 33;

    /* Determine history parameters from dilation */
    int hist_len, time_offset;
    switch (dilation) {
        case 1: hist_len = 2;  time_offset = 0; break;  /* effective H=3 */
        case 2: hist_len = 4;  time_offset = 2; break;  /* effective H=5 */
        case 5: hist_len = 10; time_offset = 6; break;  /* effective H=11 */
        default: hist_len = 2; time_offset = 0; break;
    }
    int full_time = 16; /* state->enc_conv_hist full time dimension */

    const int32_t *x1 = x, *x2 = x + 8*Win;

    /* x1: SFE (8→24) */
    int32_t *x1_sfe = (int32_t *)calloc(24*Win, sizeof(int32_t));
    SFE_fixed(x1, 8, Win, x1_sfe);

    /* x1: PC0 (24→16) */
    int32_t *x1_pc0 = (int32_t *)calloc(16*Win, sizeof(int32_t));
    P_Conv_block_0(x1_sfe, gtc_idx, pc0_w, pc0_b, pc0_bn_w, pc0_bn_b,
                   pc0_bn_m, pc0_bn_v, pc0_prelu_w, x1_pc0);

    /* x1: DD-Conv (16→16) with 3D history buffer */
    int32_t *x1_dd = (int32_t *)calloc(16*Win, sizeof(int32_t));
    dd_conv_with_hist(conv_hist, full_time, time_offset, hist_len, x1_pc0,
                      dd_w, dd_b, dd_bn_w, dd_bn_b, dd_bn_m, dd_bn_v, dd_prelu_w,
                      dilation, x1_dd);

    /* x1: PC1 (16→8) */
    int32_t *x1_pc1 = (int32_t *)calloc(8*Win, sizeof(int32_t));
    P_Conv_block_1(x1_dd, gtc_idx, pc1_w, pc1_b, pc1_bn_w, pc1_bn_b,
                   pc1_bn_m, pc1_bn_v, x1_pc1);

    /* x1: TRA (8→8) with GRU hidden state */
    int32_t *x1_tra = (int32_t *)calloc(8*Win, sizeof(int32_t));
    TRA_module(x1_pc1, h_prev, gtc_idx, tra_ih_w, tra_ih_b, tra_hh_w, tra_hh_b,
               tra_fc_w, tra_fc_b, x1_tra);

    /* Channel shuffle: y[0:2:16] = x1_tra, y[1:2:16] = x2 */
    for (int i = 0; i < 8; i++)
        for (int w = 0; w < Win; w++) {
            y[(2*i)*Win+w]   = x1_tra[i*Win+w];
            y[(2*i+1)*Win+w] = x2[i*Win+w];
        }

    free(x1_sfe); free(x1_pc0); free(x1_dd); free(x1_pc1); free(x1_tra);
}

/* ================================================================
 * GT_DeConv_module — Grouped Temporal Transposed Convolution Block
 * ================================================================
 * x_in(16,33) + x_skip(16,33) → split x1(8ch)→SFE→PDC0→DD-DeConv→PDC1→DeTRA
 * Channel shuffle: y[0:2:16]=x1_detra, y[1:2:16]=x2
 *
 * conv_hist: pointer to time-slice within state->dec_conv_hist
 *   dil=5: hist_len=10 → conv_hist points to dec_conv_hist[c*16*33 + 6*33]
 *   dil=2: hist_len=4  → conv_hist points to dec_conv_hist[c*16*33 + 2*33]
 *   dil=1: hist_len=2  → conv_hist points to dec_conv_hist[c*16*33 + 0*33]
 */

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
                      int32_t *y) {
    int Win = 33;

    int hist_len, time_offset;
    switch (dilation) {
        case 1: hist_len = 2;  time_offset = 0; break;
        case 2: hist_len = 4;  time_offset = 2; break;
        case 5: hist_len = 10; time_offset = 6; break;
        default: hist_len = 2; time_offset = 0; break;
    }
    int full_time = 16; /* state->dec_conv_hist full time dimension */

    /* Skip connection: x = x_in + x_conv */
    int32_t *x_sum = (int32_t *)calloc(16*Win, sizeof(int32_t));
    for (int i = 0; i < 16*Win; i++) x_sum[i] = sat32((int64_t)x_in[i] + x_skip[i]);

    const int32_t *x1 = x_sum, *x2 = x_sum + 8*Win;

    /* x1: SFE (8→24) */
    int32_t *x1_sfe = (int32_t *)calloc(24*Win, sizeof(int32_t));
    SFE_fixed(x1, 8, Win, x1_sfe);

    /* x1: PDC0 — Point-wise TConv (24→16) */
    int32_t *x1_pdc0 = (int32_t *)calloc(16*Win, sizeof(int32_t));
    P_DeConv_block_0(x1_sfe, gtc_idx, pc0_w, pc0_b, pc0_bn_w, pc0_bn_b,
                     pc0_bn_m, pc0_bn_v, pc0_prelu_w, x1_pdc0);

    /* x1: DD-DeConv (16→16) with 3D history */
    int32_t *x1_dd = (int32_t *)calloc(16*Win, sizeof(int32_t));
    dd_deconv_with_hist(conv_hist, full_time, time_offset, hist_len, x1_pdc0,
                        dd_w, dd_b, dd_bn_w, dd_bn_b, dd_bn_m, dd_bn_v, dd_prelu_w,
                        dilation, x1_dd);

    /* x1: PDC1 — Point-wise TConv (16→8) */
    int32_t *x1_pdc1 = (int32_t *)calloc(8*Win, sizeof(int32_t));
    P_DeConv_block_1(x1_dd, gtc_idx, pc1_w, pc1_b, pc1_bn_w, pc1_bn_b,
                     pc1_bn_m, pc1_bn_v, x1_pdc1);

    /* x1: DeTRA (8→8) */
    int32_t *x1_detra = (int32_t *)calloc(8*Win, sizeof(int32_t));
    DeTRA_module(x1_pdc1, h_prev, gtc_idx, detra_ih_w, detra_ih_b,
                 detra_hh_w, detra_hh_b, detra_fc_w, detra_fc_b, x1_detra);

    /* Channel shuffle */
    for (int i = 0; i < 8; i++)
        for (int w = 0; w < Win; w++) {
            y[(2*i)*Win+w]   = x1_detra[i*Win+w];
            y[(2*i+1)*Win+w] = x2[i*Win+w];
        }

    free(x_sum); free(x1_sfe); free(x1_pdc0); free(x1_dd); free(x1_pdc1); free(x1_detra);
}

/* ================================================================
 * Intra_RNN_module — BiGRU + FC + LN + Residual
 * ================================================================
 * x: (33, 16) s32f20 → split x1(33,8) x2(33,8) → BiGRU(nH=4) → FC → LN → +residual
 */

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
                      int32_t *y) {
    int seq_len = 33, in_dim = 8, hidden = 4, full_dim = 16;
    (void)dprnn_idx;

    /* Extract channels 0-7 and 8-15 into contiguous (seq_len, in_dim) arrays */
    int32_t *x1 = (int32_t *)calloc(seq_len*in_dim, sizeof(int32_t));
    int32_t *x2 = (int32_t *)calloc(seq_len*in_dim, sizeof(int32_t));
    for (int f = 0; f < seq_len; f++) {
        for (int c = 0; c < in_dim; c++) {
            x1[f*in_dim+c] = x[f*full_dim+c];
            x2[f*in_dim+c] = x[f*full_dim+in_dim+c];
        }
    }

    int16_t *y1_gru = (int16_t *)calloc(seq_len*2*hidden, sizeof(int16_t));
    bigru_fixed(x1, seq_len, in_dim, hidden, rnn1_ih_w, rnn1_ih_b, rnn1_hh_w, rnn1_hh_b,
                rnn1_re_ih_w, rnn1_re_ih_b, rnn1_re_hh_w, rnn1_re_hh_b, y1_gru);

    int16_t *y2_gru = (int16_t *)calloc(seq_len*2*hidden, sizeof(int16_t));
    bigru_fixed(x2, seq_len, in_dim, hidden, rnn2_ih_w, rnn2_ih_b, rnn2_hh_w, rnn2_hh_b,
                rnn2_re_ih_w, rnn2_re_ih_b, rnn2_re_hh_w, rnn2_re_hh_b, y2_gru);

    int16_t *x_gru = (int16_t *)calloc(seq_len*16, sizeof(int16_t));
    for (int f = 0; f < seq_len; f++)
        for (int h = 0; h < 2*hidden; h++) {
            x_gru[f*16+h] = y1_gru[f*(2*hidden)+h];
            x_gru[f*16+2*hidden+h] = y2_gru[f*(2*hidden)+h];
        }

    int32_t *x_fc = (int32_t *)calloc(seq_len*16, sizeof(int32_t));
    for (int f = 0; f < seq_len; f++)
        for (int co = 0; co < 16; co++) {
            int64_t acc = 0;
            for (int ci = 0; ci < 16; ci++) acc += (int64_t)x_gru[f*16+ci]*fc_w[ci*16+co];
            x_fc[f*16+co] = sat32((int64_t)((acc + 128) >> 8) + fc_b[co]);
        }

    /* Transpose x_fc from (33,16) freq-major to (16,33) channel-major for ln_fixed */
    {
        int32_t *x_fc_t = (int32_t *)calloc(16*seq_len, sizeof(int32_t));
        for (int f = 0; f < seq_len; f++)
            for (int c = 0; c < 16; c++)
                x_fc_t[c*seq_len + f] = x_fc[f*16 + c];
        ln_fixed(x_fc_t, 16, seq_len, ln_w, ln_b, -12);
        for (int f = 0; f < seq_len; f++)
            for (int c = 0; c < 16; c++)
                x_fc[f*16 + c] = x_fc_t[c*seq_len + f];
        free(x_fc_t);
    }
    for (int i = 0; i < seq_len*16; i++) y[i] = sat32((int64_t)x[i]+x_fc[i]);
    free(y1_gru); free(y2_gru); free(x_gru); free(x_fc); free(x1); free(x2);
}

/* ================================================================
 * Inter_RNN_module — GRU + FC + LN + Residual
 * ================================================================
 * x: (33, 16) → split ×2 → GRU(nH=8) → FC → LN → +residual
 */

void Inter_RNN_module(const int32_t *x, int16_t *h_prev, int dprnn_idx,
                      const int16_t *rnn1_ih_w, const int16_t *rnn1_ih_b,
                      const int16_t *rnn1_hh_w, const int16_t *rnn1_hh_b,
                      const int16_t *rnn2_ih_w, const int16_t *rnn2_ih_b,
                      const int16_t *rnn2_hh_w, const int16_t *rnn2_hh_b,
                      const int16_t *fc_w, const int32_t *fc_b,
                      const int16_t *ln_w, const int32_t *ln_b,
                      int32_t *y) {
    int seq_len = 33, in_dim = 8, hidden = 8, full_dim = 16;
    (void)dprnn_idx;

    /* Extract channels 0-7 and 8-15 into contiguous (seq_len, in_dim) arrays */
    int32_t *x1 = (int32_t *)calloc(seq_len*in_dim, sizeof(int32_t));
    int32_t *x2 = (int32_t *)calloc(seq_len*in_dim, sizeof(int32_t));
    for (int f = 0; f < seq_len; f++) {
        for (int c = 0; c < in_dim; c++) {
            x1[f*in_dim+c] = x[f*full_dim+c];
            x2[f*in_dim+c] = x[f*full_dim+in_dim+c];
        }
    }

    int16_t *y1 = (int16_t *)calloc(seq_len*hidden, sizeof(int16_t));
    gru_fixed_perframe(x1, seq_len, in_dim, hidden, rnn1_ih_w, rnn1_ih_b, rnn1_hh_w, rnn1_hh_b, y1, h_prev, full_dim);

    int16_t *y2 = (int16_t *)calloc(seq_len*hidden, sizeof(int16_t));
    gru_fixed_perframe(x2, seq_len, in_dim, hidden, rnn2_ih_w, rnn2_ih_b, rnn2_hh_w, rnn2_hh_b, y2, h_prev+hidden, full_dim);

    int16_t *x_gru = (int16_t *)calloc(seq_len*16, sizeof(int16_t));
    for (int f = 0; f < seq_len; f++)
        for (int h = 0; h < hidden; h++) {
            x_gru[f*16+h] = y1[f*hidden+h];
            x_gru[f*16+hidden+h] = y2[f*hidden+h];
        }

    int32_t *x_fc = (int32_t *)calloc(seq_len*16, sizeof(int32_t));
    for (int f = 0; f < seq_len; f++)
        for (int co = 0; co < 16; co++) {
            int64_t acc = 0;
            for (int ci = 0; ci < 16; ci++) acc += (int64_t)x_gru[f*16+ci]*fc_w[ci*16+co];
            x_fc[f*16+co] = sat32((int64_t)((acc + 128) >> 8) + fc_b[co]);
        }

    /* Transpose x_fc from (33,16) freq-major to (16,33) channel-major for ln_fixed */
    {
        int32_t *x_fc_t = (int32_t *)calloc(16*seq_len, sizeof(int32_t));
        for (int f = 0; f < seq_len; f++)
            for (int c = 0; c < 16; c++)
                x_fc_t[c*seq_len + f] = x_fc[f*16 + c];
        ln_fixed(x_fc_t, 16, seq_len, ln_w, ln_b, -12);
        for (int f = 0; f < seq_len; f++)
            for (int c = 0; c < 16; c++)
                x_fc[f*16 + c] = x_fc_t[c*seq_len + f];
        free(x_fc_t);
    }
    for (int i = 0; i < seq_len*16; i++) y[i] = sat32((int64_t)x[i]+x_fc[i]);
    free(y1); free(y2); free(x_gru); free(x_fc); free(x1); free(x2);
}

/* ================================================================
 * GDPRNN_module — Intra-RNN → Inter-RNN
 * ================================================================
 * x: (16, 33) → transpose → Intra(33,16) → Inter(33,16) → transpose → (16,33)
 */

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
                   int32_t *y) {
    int32_t *xt = (int32_t *)calloc(33*16, sizeof(int32_t));
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 33; j++) xt[j*16+i] = x[i*33+j];

    int32_t *yi = (int32_t *)calloc(33*16, sizeof(int32_t));
    Intra_RNN_module(xt, dprnn_idx, rnn1_ih_w, rnn1_ih_b, rnn1_hh_w, rnn1_hh_b,
                     rnn1_re_ih_w, rnn1_re_ih_b, rnn1_re_hh_w, rnn1_re_hh_b,
                     rnn2_ih_w, rnn2_ih_b, rnn2_hh_w, rnn2_hh_b,
                     rnn2_re_ih_w, rnn2_re_ih_b, rnn2_re_hh_w, rnn2_re_hh_b,
                     intra_fc_w, intra_fc_b, intra_ln_w, intra_ln_b, yi);

    int32_t *ye = (int32_t *)calloc(33*16, sizeof(int32_t));
    Inter_RNN_module(yi, inter_prev, dprnn_idx,
                     inter_rnn1_ih_w, inter_rnn1_ih_b,
                     inter_rnn1_hh_w, inter_rnn1_hh_b,
                     inter_rnn2_ih_w, inter_rnn2_ih_b,
                     inter_rnn2_hh_w, inter_rnn2_hh_b,
                     inter_fc_w, inter_fc_b, inter_ln_w, inter_ln_b, ye);

    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 33; j++) y[i*33+j] = ye[j*16+i];
    free(xt); free(yi); free(ye);
}

/* ================================================================
 * denoise_infer_frame — Single-frame full pipeline stub
 * ================================================================
 * The full wired version is in test_denoise_full.c.
 * This stub allows denoise_fp.c to compile without the weight header.
 */

/* Weak symbol — can be overridden by test_denoise_full.c which includes weights */
__attribute__((weak))
void denoise_infer_frame(const float *real_in, const float *imag_in,
                       denoise_state_t *state,
                       const uint16_t *erbfc_w, const uint16_t *ierbfc_w,
                       int32_t *crm_out) {
    (void)real_in; (void)imag_in; (void)state; (void)erbfc_w; (void)ierbfc_w; (void)crm_out;
}

/* End of denoise_fp.c */