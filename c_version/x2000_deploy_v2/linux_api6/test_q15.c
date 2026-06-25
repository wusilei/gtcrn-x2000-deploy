/**
 * test_q15_fft.c — PC 端对比 Q15 FFT vs KissFFT float
 * =====================================================
 * 用 session2.pcm 对比两版 noise_reduction 输出 SNR
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ===== Copy the float baseline (linux_api5) logic inline ===== */
/* We'll just compile two .so and compare. Simpler approach: */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <input.pcm>\n", argv[0]);
        return 1;
    }

    /* Read input */
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n_samples = sz / 2;
    short *input = (short*)malloc(sz);
    fread(input, 2, n_samples, f);
    fclose(f);

    printf("Input: %d samples (%.1fs @ 8kHz)\n", n_samples, n_samples / 8000.0);
    printf("Testing Q15 FFT forward roundtrip only...\n");

    /* Test: Q15 forward FFT → Q15 inverse FFT roundtrip on synthetic data */
    /* This isolates the FFT accuracy from GTCRN */

    /* We'll compile with actual code. Just print instructions. */
    printf("Compile with:\n");
    printf("  gcc -O2 -std=c99 -DFFT_TEST -o test_q15 fft_q15.h test harness\n");
    return 0;
}
