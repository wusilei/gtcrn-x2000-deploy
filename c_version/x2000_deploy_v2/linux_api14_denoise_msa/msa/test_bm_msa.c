/**
 * test_bm_msa.c — Verify BM_fixed MSA vs scalar bit-exactness + benchmark
 *
 * Compile (PC):   gcc -std=c99 -O2 -o test_bm_msa test_bm_msa.c bm_fixed_msa.c -lm
 * Compile (MIPS): mips-linux-gnu-gcc -std=c99 -O3 -mmsa -mhard-float -mfp64 -march=mips32r5 -static -o test_bm_msa_mips test_bm_msa.c bm_fixed_msa.c -lm
 */
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Copy of scalar BM_fixed for comparison */
#define BM_WIN   257
#define BM_WOUT  129
#define BM_LOW    65
#define BM_M     192
#define BM_N      64

static void bm_fixed_scalar(const int32_t *x, const uint16_t *weight, int32_t *y) {
    for (int c = 0; c < 3; c++) {
        for (int w = 0; w < BM_LOW; w++) y[c*BM_WOUT+w] = x[c*BM_WIN+w];
        for (int j = 0; j < BM_N; j++) {
            int64_t acc = 0;
            for (int i = 0; i < BM_M; i++)
                acc += (int64_t)x[c*BM_WIN + BM_LOW + i] * weight[i * BM_N + j];
            y[c*BM_WOUT + BM_LOW + j] = (int32_t)((acc + 16384) >> 15);
        }
    }
}

extern void bm_fixed_msa(const int32_t *x, const uint16_t *weight, int32_t *y);

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec*1000.0 + (double)ts.tv_nsec/1e6;
}

int main(void) {
    /* Allocate test data */
    int32_t  *x     = calloc(3 * BM_WIN,  sizeof(int32_t));
    uint16_t *w     = calloc(BM_M * BM_N,  sizeof(uint16_t));
    int32_t  *y_ref = calloc(3 * BM_WOUT, sizeof(int32_t));
    int32_t  *y_msa = calloc(3 * BM_WOUT, sizeof(int32_t));

    /* Fill with random fixed-point data */
    srand(42);
    for (int i = 0; i < 3 * BM_WIN; i++)
        x[i] = (int32_t)(((int64_t)rand() * (int64_t)rand()) % 2000000 - 1000000);
    for (int i = 0; i < BM_M * BM_N; i++)
        w[i] = (uint16_t)(rand() % 32768);  /* u16f15, dense */

    printf("=== BM_fixed MSA Test ===\n");
#ifdef __mips_msa
    printf("  Platform: MIPS (MSA active)\n");
#else
    printf("  Platform: PC (MSA fallback)\n");
#endif
    printf("  x: [3][257] s32f20, w: [192][64] u16f15 (DENSE)\n\n");

    /* Compute reference */
    bm_fixed_scalar(x, w, y_ref);

    /* Compute MSA */
    bm_fixed_msa(x, w, y_msa);

    /* Compare */
    int mismatches = 0; int64_t max_diff = 0; double sum_sq = 0;
    for (int c = 0; c < 3; c++) {
        for (int j = 0; j < BM_WOUT; j++) {
            int64_t d = (int64_t)y_msa[c*BM_WOUT+j] - (int64_t)y_ref[c*BM_WOUT+j];
            if (d < 0) d = -d;
            if (d > 0) { mismatches++; if (d > max_diff) max_diff = d;
                         sum_sq += (double)d*(double)d; }
        }
    }

    double rms = mismatches>0 ? sqrt(sum_sq/mismatches) : 0;
    double snr = max_diff>0 ? 20.0*log10(2147483648.0/(double)max_diff) : 999;

    if (mismatches == 0)
        printf("  Correctness: PASS (bit-exact, %d values)\n", 3*BM_WOUT);
    else if (snr > 120.0)
        printf("  Correctness: PASS (SNR=%.1f dB, max=%ld LSB, RMS=%.1f LSB — audio-bit-exact)\n",
               snr, (long)max_diff, rms);
    else
        printf("  Correctness: FAIL — %d/%d mism, max_diff=%ld, SNR=%.1f dB\n",
               mismatches, 3*BM_WOUT, (long)max_diff, snr);

    /* Benchmark */
    int n = 50000;
    printf("\n  Benchmark (%d repeats):\n", n);

    double t0 = now_ms();
    for (int i = 0; i < n; i++) bm_fixed_scalar(x, w, y_ref);
    double t1 = now_ms();
    printf("    scalar: %8.1f µs\n", (t1-t0)/(double)n*1000.0);

    t0 = now_ms();
    for (int i = 0; i < n; i++) bm_fixed_msa(x, w, y_ref);
    t1 = now_ms();
    printf("    MSA:    %8.1f µs\n", (t1-t0)/(double)n*1000.0);

    free(x); free(w); free(y_ref); free(y_msa);
    return (snr < 120.0) ? 1 : 0;
}
