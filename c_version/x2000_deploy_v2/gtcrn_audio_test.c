/**
 * gtcrn_audio_test.c — GTCRN Audio Processing Test for X2000
 * ============================================================
 * Reads pre-computed STFT frames (float32 binary: N_frames × 514),
 * processes each frame through GTCRN, outputs CRM (int32 s32f20).
 *
 * Usage:
 *   ./gtcrn_audio_test <input_stft.bin> <output_crm.bin> <num_frames>
 *
 * Input:  [real_0..real_256, imag_0..imag_256] × N_frames  (float32)
 * Output: [I_0..I_256, Q_0..Q_256] × N_frames              (int32 s32f20)
 */
#include "gtcrn_matlab_weights.h"
#include "gtcrn_fp.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_stft.bin> <output_crm.bin> <num_frames>\n", argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];
    int num_frames = atoi(argv[3]);

    if (num_frames <= 0) {
        fprintf(stderr, "Invalid num_frames: %d\n", num_frames);
        return 1;
    }

    /* Open input file */
    FILE *fin = fopen(input_path, "rb");
    if (!fin) {
        perror("fopen input");
        return 1;
    }

    /* Open output file */
    FILE *fout = fopen(output_path, "wb");
    if (!fout) {
        perror("fopen output");
        fclose(fin);
        return 1;
    }

    /* Initialize state */
    gtcrn_state_t state;
    gtcrn_state_init(&state);

    /* Buffers */
    float frame_in[514];       /* 257 real + 257 imag */
    int32_t crm_out[2 * 257];  /* 257 I + 257 Q, s32f20 */

    clock_t start = clock();
    double total_ms = 0.0;

    for (int f = 0; f < num_frames; f++) {
        /* Read one STFT frame */
        size_t nread = fread(frame_in, sizeof(float), 514, fin);
        if (nread != 514) {
            fprintf(stderr, "Frame %d: short read (%zu/514)\n", f, nread);
            break;
        }

        /* Process through GTCRN */
        gtcrn_infer_frame(frame_in, frame_in + 257, &state,
                          erb_erb_fc_weight, erb_ierb_fc_weight,
                          crm_out);

        /* Write CRM output */
        fwrite(crm_out, sizeof(int32_t), 2 * 257, fout);
    }

    clock_t end = clock();
    total_ms = 1000.0 * (double)(end - start) / (double)CLOCKS_PER_SEC;

    fclose(fin);
    fclose(fout);

    printf("GTCRN Audio Test Complete\n");
    printf("  Frames:       %d\n", num_frames);
    printf("  Total time:   %.1f ms\n", total_ms);
    printf("  Per frame:    %.2f ms\n", total_ms / num_frames);
    printf("  Real-time:    %s\n", (total_ms / num_frames) < 32.0 ? "YES" : "NO");

    return 0;
}
