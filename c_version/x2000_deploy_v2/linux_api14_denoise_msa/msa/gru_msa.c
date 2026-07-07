/**
 * gru_msa.c — MSA-accelerated GRU/BiGRU gate dot products
 *
 * Accelerates the GRU gate input-to-hidden and hidden-to-hidden
 * dot products using MSA madd_q_w (4 lanes × 1 multiply-accumulate).
 *
 * Q-format: x Q20 × w Q12 → product Q32.  Shift >>22 → Q10.
 * MSA: w_Q30 = w << 18 (Q12→Q30, safe: max 4096<<18 = 1G < INT32_MAX)
 *      madd_q_w gives Q19.  >>9 → Q10.  Same as scalar.
 */

#include <stdint.h>
#ifdef __mips_msa
#include <msa.h>
#endif

/* ── MSA dot product: y[h..h+3] = Σ_i xv[i] * w[i*stride + h..h+3] ───
 *
 * xv       : int32_t array (Q20), length n
 * w        : int16_t array (Q12), shape [n][stride]
 * stride   : int, w row stride in int16 elements (typically 3*hidden_dim)
 * n        : int, number of input elements
 * h        : int, starting hidden dim (must be multiple of 4, h+3 < stride)
 * y_out    : int32_t[4], Q19 output (caller shifts >>9 → Q10)
 *
 * w access is strided: w[i*stride + h..h+3] loads 4 contiguous int16 values.
 * BUT: w[i*stride + h] where stride can be large means ld_w may read 4
 * int32 values from addresses that span beyond the row. We use ld_w with
 * unpack (same as bm_fixed_msa) for safe 4-element contiguous loads.
 */

#ifdef __mips_msa
static void msa_dotp_ih_q19(
    const int32_t *xv, const int16_t *w, int stride, int n, int h,
    int32_t *y_out)
{
    v4i32 acc = __builtin_msa_fill_w(0);
    for (int i = 0; i < n; i++) {
        v4i32 xs = __builtin_msa_fill_w(xv[i]);
        /* Load 4 contiguous int16 weights from w[i*stride + h] */
        const int16_t *wp = &w[i * stride + h];
        v4i32 w32 = __builtin_msa_ld_w((void*)wp, 0);
        v4i32 w_lo = w32 & __builtin_msa_fill_w(0x0000FFFF);
        v4i32 w_hi = __builtin_msa_srli_w(w32, 16);
        v4i32 w_q12 = (v4i32)__builtin_msa_ilvr_w(w_hi, w_lo);
        v4i32 w_q30 = __builtin_msa_slli_w(w_q12, 18);  /* Q12→Q30 */
        acc = __builtin_msa_madd_q_w(acc, xs, w_q30);
    }
    y_out[0] = __builtin_msa_copy_s_w(acc, 0);
    y_out[1] = __builtin_msa_copy_s_w(acc, 1);
    y_out[2] = __builtin_msa_copy_s_w(acc, 2);
    y_out[3] = __builtin_msa_copy_s_w(acc, 3);
}

/* Same for hidden-to-hidden: hh dimension = hidden_dim, weight is Q12 */
static void msa_dotp_hh_q19(
    const int16_t *xv_q15, const int16_t *w, int stride, int n, int h,
    int32_t *y_out)
{
    v4i32 acc = __builtin_msa_fill_w(0);
    for (int i = 0; i < n; i++) {
        /* xv is Q15 (int16), extend to int32 */
        int32_t x32 = (int32_t)xv_q15[i];
        v4i32 xs = __builtin_msa_fill_w(x32);
        const int16_t *wp = &w[i * stride + h];
        v4i32 w32 = __builtin_msa_ld_w((void*)wp, 0);
        v4i32 w_lo = w32 & __builtin_msa_fill_w(0x0000FFFF);
        v4i32 w_hi = __builtin_msa_srli_w(w32, 16);
        v4i32 w_q12 = (v4i32)__builtin_msa_ilvr_w(w_hi, w_lo);
        v4i32 w_q30 = __builtin_msa_slli_w(w_q12, 18);
        acc = __builtin_msa_madd_q_w(acc, xs, w_q30);
    }
    y_out[0] = __builtin_msa_copy_s_w(acc, 0);
    y_out[1] = __builtin_msa_copy_s_w(acc, 1);
    y_out[2] = __builtin_msa_copy_s_w(acc, 2);
    y_out[3] = __builtin_msa_copy_s_w(acc, 3);
}
#endif /* __mips_msa */
