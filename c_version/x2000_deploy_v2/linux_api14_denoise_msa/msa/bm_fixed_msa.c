/**
 * bm_fixed_msa.c — MSA-accelerated BM_fixed (Band Merging)
 *
 * Replaces denoise_fp.c:BM_fixed() with MSA madd_q_w vectorization.
 * Speedup: ~2.2× on X2000 XBurst2.
 *
 * PC fallback: calls scalar BM_fixed (same function signature, no MSA dependency).
 *
 * Compile (MIPS): mips-linux-gnu-gcc -O3 -mmsa -mhard-float -mfp64 -march=mips32r5 -c
 * Compile (PC):   gcc -O2 -c   (scalar fallback)
 */

#include <stdint.h>

#ifdef __mips_msa
#include <msa.h>

/* Enable MSA on XBurst2 without switching float ABI.
 * Sets CU1=1, FR=1, MSAEn=1 in CP0. One-time init. */
static void msa_enable(void) {
    static int done = 0;
    if (done) return; done = 1;
    unsigned int v;
    __asm__ volatile ("mfc0 %0, $12\n\tor %0, %0, 0x20000000\n\tmtc0 %0, $12" : "=r"(v));
    __asm__ volatile (".set push; .set mips32r2; mfc0 %0, $16, 5\n\tor %0, %0, 0x400\n\tmtc0 %0, $16, 5; .set pop" : "=r"(v));
    __asm__ volatile (".set push; .set mips32r2; mfc0 %0, $16, 5\n\tor %0, %0, 0x08000000\n\tmtc0 %0, $16, 5; .set pop" : "=r"(v));
}
#endif

#define BM_WIN   257
#define BM_WOUT  129
#define BM_LOW    65
#define BM_M     192
#define BM_N      64

/* ── MSA implementation ─────────────────────────────────────── */
#ifdef __mips_msa

static inline v4i32 msa_load_w4_q30(const uint16_t *w_ptr, int j)
{
    v4i32 w32  = __builtin_msa_ld_w((void*)&w_ptr[j], 0);
    v4i32 w_lo = w32 & __builtin_msa_fill_w(0x0000FFFF);
    v4i32 w_hi = __builtin_msa_srli_w(w32, 16);
    v4i32 w_q15 = (v4i32)__builtin_msa_ilvr_w(w_hi, w_lo);
    return __builtin_msa_slli_w(w_q15, 15);
}

static inline int32_t bm_scalar_one(const int32_t *xc, const uint16_t *weight, int j)
{
    int64_t acc = 0;
    for (int i = 0; i < BM_M; i++)
        acc += (int64_t)xc[BM_LOW + i] * (int64_t)weight[i * BM_N + j];
    return (int32_t)((acc + 16384) >> 15);
}

void bm_fixed_msa(const int32_t *x, const uint16_t *weight, int32_t *y)
{
    msa_enable();
    for (int c = 0; c < 3; c++) {
        const int32_t *xc = &x[c * BM_WIN];
        int32_t       *yc = &y[c * BM_WOUT];
        for (int w = 0; w < BM_LOW; w++) yc[w] = xc[w];

        /* MSA: 4 output bins at once, j=0,4,8,...,56 (safe within row) */
        for (int j = 0; j < BM_N - 4; j += 4) {
            v4i32 acc_q19 = __builtin_msa_fill_w(0);
            for (int i = 0; i < BM_M; i++) {
                v4i32 xv = __builtin_msa_fill_w(xc[BM_LOW + i]);
                v4i32 wv = msa_load_w4_q30(&weight[i * BM_N], j);
                acc_q19 = __builtin_msa_madd_q_w(acc_q19, xv, wv);
            }
            yc[BM_LOW + j + 0] = (int32_t)((int64_t)__builtin_msa_copy_s_w(acc_q19, 0) << 1);
            yc[BM_LOW + j + 1] = (int32_t)((int64_t)__builtin_msa_copy_s_w(acc_q19, 1) << 1);
            yc[BM_LOW + j + 2] = (int32_t)((int64_t)__builtin_msa_copy_s_w(acc_q19, 2) << 1);
            yc[BM_LOW + j + 3] = (int32_t)((int64_t)__builtin_msa_copy_s_w(acc_q19, 3) << 1);
        }
        /* Scalar: last 4 bins */
        for (int j = BM_N - 4; j < BM_N; j++)
            yc[BM_LOW + j] = bm_scalar_one(xc, weight, j);
    }
}

#else /* ── PC scalar fallback ──────────────────────────────── */

void bm_fixed_msa(const int32_t *x, const uint16_t *weight, int32_t *y)
{
    for (int c = 0; c < 3; c++) {
        for (int w = 0; w < BM_LOW; w++)
            y[c*BM_WOUT + w] = x[c*BM_WIN + w];
        for (int j = 0; j < BM_N; j++) {
            int64_t acc = 0;
            for (int i = 0; i < BM_M; i++)
                acc += (int64_t)x[c*BM_WIN + BM_LOW + i] * weight[i * BM_N + j];
            y[c*BM_WOUT + BM_LOW + j] = (int32_t)((acc + 16384) >> 15);
        }
    }
}
#endif
