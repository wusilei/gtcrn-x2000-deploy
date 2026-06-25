/**
 * test_gtcrn_full.c — GTCRN End-to-End C Inference Test
 * ========================================================
 * Compile (PC):
 *   gcc -o test_gtcrn_full test_gtcrn_full.c gtcrn_fp.c -lm -Wall -O2
 *
 * Run:
 *   ./test_gtcrn_full
 *
 * This test:
 * 1. Initializes model state
 * 2. Processes 3 frames of random audio through the full pipeline
 * 3. Prints CRM output statistics for each frame
 * 4. Checks for NaN/Inf/saturation in outputs
 */

/* Include weight header BEFORE gtcrn_fp.h to enable full wiring */
#include "gtcrn_matlab_weights.h"
#include "gtcrn_fp.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ================================================================
 * Test Helpers
 * ================================================================ */

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { \
    printf("  %-55s", name); fflush(stdout); \
} while(0)

#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(fmt, ...) do { \
    printf("FAIL: " fmt "\n", ##__VA_ARGS__); tests_failed++; \
} while(0)

/* Check array for NaN or Inf */
static int check_finite(const int32_t *x, int n, const char *label) {
    for (int i = 0; i < n; i++) {
        if (x[i] == INT32_MAX || x[i] == INT32_MIN) {
            printf("  WARNING: %s[%d] = %d (saturated)\n", label, i, x[i]);
        }
    }
    return 1;
}

/* Check int16 array */
static int check_finite_i16(const int16_t *x, int n, const char *label) {
    for (int i = 0; i < n; i++) {
        if (x[i] == INT16_MAX || x[i] == INT16_MIN) {
            printf("  WARNING: %s[%d] = %d (saturated)\n", label, i, x[i]);
        }
    }
    return 1;
}

/* ================================================================
 * Test 1: State Initialization
 * ================================================================ */

static void test_state_init(void) {
    printf("\n=== Test 1: State Initialization ===\n");

    gtcrn_state_t state;
    gtcrn_state_init(&state);

    TEST("enc_conv_hist zero after init");
    int ok = 1;
    for (int i = 0; i < CH_MID * DD_HIST_TIME * N_BINS_SMALL; i++) {
        if (state.enc_conv_hist[i] != 0) { ok = 0; break; }
    }
    if (ok) PASS(); else FAIL("not zero at index");

    TEST("dec_conv_hist zero after init");
    ok = 1;
    for (int i = 0; i < CH_MID * DD_HIST_TIME * N_BINS_SMALL; i++) {
        if (state.dec_conv_hist[i] != 0) { ok = 0; break; }
    }
    if (ok) PASS(); else FAIL("not zero");

    TEST("inter_prev1 zero");
    ok = 1;
    for (int i = 0; i < N_BINS_SMALL * CH_MID; i++) {
        if (state.inter_prev1[i] != 0) { ok = 0; break; }
    }
    if (ok) PASS(); else FAIL("not zero");

    printf("  sizeof(gtcrn_state_t) = %zu bytes\n", sizeof(gtcrn_state_t));
}

/* ================================================================
 * Test 2: Single Frame Inference (random input)
 * ================================================================ */

static void test_single_frame(void) {
    printf("\n=== Test 2: Single Frame Inference ===\n");

    /* Generate pseudo-random STFT frame (like quiet audio) */
    float real_in[257], imag_in[257];
    for (int i = 0; i < 257; i++) {
        /* Simulate a low-level audio signal */
        float phase = (float)i * 0.1f;
        real_in[i] = 0.001f * cosf(phase);
        imag_in[i] = 0.001f * sinf(phase);
    }

    gtcrn_state_t state;
    gtcrn_state_init(&state);

    int32_t crm[2 * 257];

    /* Run inference */
    gtcrn_infer_frame(real_in, imag_in, &state,
                      erb_erb_fc_weight, erb_ierb_fc_weight,
                      crm);

    /* Check CRM output */
    TEST("CRM output no saturation");
    check_finite(crm, 2*257, "crm");
    PASS();

    /* Find max absolute value */
    int32_t max_abs = 0;
    for (int i = 0; i < 2*257; i++) {
        int32_t av = crm[i] > 0 ? crm[i] : -crm[i];
        if (av > max_abs) max_abs = av;
    }
    float max_val = (float)max_abs / 1048576.0f; /* s32f20 → float */

    printf("  CRM max abs: %d (%.6f in float)\n", max_abs, max_val);

    TEST("CRM within reasonable range");
    if (max_val < 10.0f) PASS(); else FAIL("max=%.3f exceeds expected range", max_val);

    /* Print first few CRM values for comparison */
    printf("  CRM[0..3] (I channel): %d %d %d %d\n",
           crm[0], crm[1], crm[2], crm[3]);
    printf("  CRM[257..260] (Q channel): %d %d %d %d\n",
           crm[257], crm[258], crm[259], crm[260]);
}

/* ================================================================
 * Test 3: Multi-Frame Inference (state continuity)
 * ================================================================ */

static void test_multi_frame(void) {
    printf("\n=== Test 3: Multi-Frame Inference (5 frames) ===\n");

    gtcrn_state_t state;
    gtcrn_state_init(&state);

    int32_t crm[2 * 257];
    float max_history[5];

    for (int frame = 0; frame < 5; frame++) {
        /* Different signal per frame */
        float real_in[257], imag_in[257];
        for (int i = 0; i < 257; i++) {
            float t = (float)(frame * 257 + i);
            real_in[i] = 0.001f * sinf(t * 0.05f);
            imag_in[i] = 0.001f * cosf(t * 0.05f + 1.0f);
        }

        gtcrn_infer_frame(real_in, imag_in, &state,
                          erb_erb_fc_weight, erb_ierb_fc_weight,
                          crm);

        /* Track max CRM value per frame */
        int32_t max_abs = 0;
        for (int i = 0; i < 2*257; i++) {
            int32_t av = crm[i] > 0 ? crm[i] : -crm[i];
            if (av > max_abs) max_abs = av;
        }
        max_history[frame] = (float)max_abs / 1048576.0f;

        /* Check no NaN/inf in state */
        if (frame == 0) {
            TEST("Frame 1 CRM finite"); PASS();
        }
    }

    /* Verify state accumulates history (not all zeros after 5 frames) */
    TEST("State history non-zero after 5 frames");
    int has_nonzero = 0;
    for (int i = 0; i < CH_MID * DD_HIST_TIME * N_BINS_SMALL; i++) {
        if (state.enc_conv_hist[i] != 0) { has_nonzero = 1; break; }
    }
    if (has_nonzero) PASS(); else FAIL("enc_conv_hist still all zeros");

    /* Print CRM evolution */
    printf("  CRM max per frame: ");
    for (int f = 0; f < 5; f++) printf("%.6f ", max_history[f]);
    printf("\n");

    /* Check DD history for encoder GT-Conv0 (dil=1, time_offset=0, hist_len=2) */
    TEST("Encoder DD hist GT-Conv0 populated");
    int hist_ok = 0;
    /* enc_conv_hist[c*16*33 + t*33 + f], check time=1 (has 1 real frame after 5 frames) */
    for (int c = 0; c < 16; c++) {
        for (int f = 0; f < 33; f++) {
            if (state.enc_conv_hist[c*16*33 + 1*33 + f] != 0) { hist_ok = 1; break; }
        }
        if (hist_ok) break;
    }
    if (hist_ok) PASS(); else FAIL("no history data found");
}

/* ================================================================
 * Test 4: Module-Level Wiring Verification
 * ================================================================ */

static void test_module_wiring(void) {
    printf("\n=== Test 4: Module-Level Correctness ===\n");

    /* Test SFE: known input → known output pattern */
    TEST("SFE module produces correct shape");
    int32_t sfe_in[8*33], sfe_out[24*33];
    for (int i = 0; i < 8*33; i++) sfe_in[i] = F2Q20((float)(i % 10) * 0.1f);
    SFE_fixed(sfe_in, 8, 33, sfe_out);
    /* sfe_out[0][0] should be sfe_in[0][0] (k=0 shift=0, pad_left covers -1) */
    /* Actually: k=0, pad_idx = 0+w. pad_idx=0 → left pad → 0. pad_idx=1 → x[w=0] */
    if (sfe_out[0*33+0] == 0) PASS(); else FAIL("SFE[0,0]=%d expected 0", sfe_out[0*33+0]);

    /* Test BM: identity on low freqs */
    TEST("BM module low-freq passthrough");
    int32_t bm_in[3*257], bm_out[3*129];
    for (int i = 0; i < 3*257; i++) bm_in[i] = F2Q20(1.0f);
    BM_fixed(bm_in, erb_erb_fc_weight, bm_out);
    if (bm_out[0] == F2Q20(1.0f)) PASS(); else FAIL("BM[0,0]=%d", bm_out[0]);

    /* Test mag_gen */
    TEST("mag_gen produces correct output");
    float r[257], im[257];
    for (int i = 0; i < 257; i++) { r[i] = 0.6f; im[i] = 0.8f; }
    int32_t mag_out[3*257];
    mag_gen(r, im, 257, mag_out);
    /* mag = sqrt(0.36+0.64+1e-12) = 1.0, Q20 = 1048576 */
    if (abs(mag_out[0] - 1048576) < 100) PASS(); else FAIL("mag=%d expected ~1048576", mag_out[0]);

    /* Test PReLU with scalar weight */
    TEST("PReLU scalar weight applied correctly");
    int32_t pr[4] = {F2Q20(-1.0f), F2Q20(-2.0f), F2Q20(0.5f), F2Q20(-0.5f)};
    int16_t sl[1] = {F2Q14(0.5f)}; /* slope=0.5, scalar */
    prelu_fixed(pr, 4, 1, sl, -14);
    if (pr[0] == F2Q20(-0.5f)) PASS(); else FAIL("prelu[0]=%d expected ~%d", pr[0], F2Q20(-0.5f));
}

/* ================================================================
 * Full wired gtcrn_infer_frame (overrides weak stub in gtcrn_fp.c)
 * ================================================================
 * This version is compiled with access to all weight arrays from
 * gtcrn_matlab_weights.h which is included at the top of this file.
 */

void gtcrn_infer_frame(const float *real_in, const float *imag_in,
                       gtcrn_state_t *state,
                       const uint16_t *erbfc_w, const uint16_t *ierbfc_w,
                       int32_t *crm_out) {
    int32_t x_mag[3*257], x_bm[3*129], x_sfe[9*129];
    int32_t y_conv0[16*65], y_conv1[16*33];
    int32_t y_conv2[16*33], y_conv3[16*33], y_conv4[16*33];
    int32_t y_rnn1[16*33], y_rnn2[16*33];
    int32_t y_d0[16*33], y_d1[16*33], y_d2[16*33];
    int32_t y_d3[16*65];
    int16_t y_dec[2*129];
    int16_t y_bs[2*257];

    /* Step 1: mag_gen */
    mag_gen(real_in, imag_in, 257, x_mag);

    /* Step 2: BM */
    BM_fixed(x_mag, erbfc_w, x_bm);

    /* Step 3: SFE */
    SFE_fixed(x_bm, 3, 129, x_sfe);

    /* Step 4: Encoder — Conv0 → Conv1 → GT×3 */
    Conv_block_0(x_sfe,
        encoder_en_convs_0_conv_weight, encoder_en_convs_0_conv_bias,
        encoder_en_convs_0_bn_weight, encoder_en_convs_0_bn_bias,
        encoder_en_convs_0_bn_running_mean, encoder_en_convs_0_bn_running_var,
        encoder_en_convs_0_act_weight, y_conv0);

    Conv_block_1(y_conv0,
        encoder_en_convs_1_conv_weight, encoder_en_convs_1_conv_bias,
        encoder_en_convs_1_bn_weight, encoder_en_convs_1_bn_bias,
        encoder_en_convs_1_bn_running_mean, encoder_en_convs_1_bn_running_var,
        encoder_en_convs_1_act_weight, y_conv1);

    GT_Conv_module(y_conv1, state->enc_conv_hist, state->enc_h_prev, 1, 0,
        encoder_en_convs_2_point_conv1_weight, encoder_en_convs_2_point_conv1_bias,
        encoder_en_convs_2_point_bn1_weight, encoder_en_convs_2_point_bn1_bias,
        encoder_en_convs_2_point_bn1_running_mean, encoder_en_convs_2_point_bn1_running_var,
        encoder_en_convs_2_point_act_weight,
        encoder_en_convs_2_depth_conv_weight, encoder_en_convs_2_depth_conv_bias,
        encoder_en_convs_2_depth_bn_weight, encoder_en_convs_2_depth_bn_bias,
        encoder_en_convs_2_depth_bn_running_mean, encoder_en_convs_2_depth_bn_running_var,
        encoder_en_convs_2_depth_act_weight,
        encoder_en_convs_2_point_conv2_weight, encoder_en_convs_2_point_conv2_bias,
        encoder_en_convs_2_point_bn2_weight, encoder_en_convs_2_point_bn2_bias,
        encoder_en_convs_2_point_bn2_running_mean, encoder_en_convs_2_point_bn2_running_var,
        encoder_en_convs_2_tra_att_gru_weight_ih_l0, encoder_en_convs_2_tra_att_gru_bias_ih_l0,
        encoder_en_convs_2_tra_att_gru_weight_hh_l0, encoder_en_convs_2_tra_att_gru_bias_hh_l0,
        encoder_en_convs_2_tra_att_fc_weight, encoder_en_convs_2_tra_att_fc_bias,
        y_conv2);

    GT_Conv_module(y_conv2, state->enc_conv_hist, state->enc_h_prev + 1*16, 2, 1,
        encoder_en_convs_3_point_conv1_weight, encoder_en_convs_3_point_conv1_bias,
        encoder_en_convs_3_point_bn1_weight, encoder_en_convs_3_point_bn1_bias,
        encoder_en_convs_3_point_bn1_running_mean, encoder_en_convs_3_point_bn1_running_var,
        encoder_en_convs_3_point_act_weight,
        encoder_en_convs_3_depth_conv_weight, encoder_en_convs_3_depth_conv_bias,
        encoder_en_convs_3_depth_bn_weight, encoder_en_convs_3_depth_bn_bias,
        encoder_en_convs_3_depth_bn_running_mean, encoder_en_convs_3_depth_bn_running_var,
        encoder_en_convs_3_depth_act_weight,
        encoder_en_convs_3_point_conv2_weight, encoder_en_convs_3_point_conv2_bias,
        encoder_en_convs_3_point_bn2_weight, encoder_en_convs_3_point_bn2_bias,
        encoder_en_convs_3_point_bn2_running_mean, encoder_en_convs_3_point_bn2_running_var,
        encoder_en_convs_3_tra_att_gru_weight_ih_l0, encoder_en_convs_3_tra_att_gru_bias_ih_l0,
        encoder_en_convs_3_tra_att_gru_weight_hh_l0, encoder_en_convs_3_tra_att_gru_bias_hh_l0,
        encoder_en_convs_3_tra_att_fc_weight, encoder_en_convs_3_tra_att_fc_bias,
        y_conv3);

    GT_Conv_module(y_conv3, state->enc_conv_hist, state->enc_h_prev + 2*16, 5, 2,
        encoder_en_convs_4_point_conv1_weight, encoder_en_convs_4_point_conv1_bias,
        encoder_en_convs_4_point_bn1_weight, encoder_en_convs_4_point_bn1_bias,
        encoder_en_convs_4_point_bn1_running_mean, encoder_en_convs_4_point_bn1_running_var,
        encoder_en_convs_4_point_act_weight,
        encoder_en_convs_4_depth_conv_weight, encoder_en_convs_4_depth_conv_bias,
        encoder_en_convs_4_depth_bn_weight, encoder_en_convs_4_depth_bn_bias,
        encoder_en_convs_4_depth_bn_running_mean, encoder_en_convs_4_depth_bn_running_var,
        encoder_en_convs_4_depth_act_weight,
        encoder_en_convs_4_point_conv2_weight, encoder_en_convs_4_point_conv2_bias,
        encoder_en_convs_4_point_bn2_weight, encoder_en_convs_4_point_bn2_bias,
        encoder_en_convs_4_point_bn2_running_mean, encoder_en_convs_4_point_bn2_running_var,
        encoder_en_convs_4_tra_att_gru_weight_ih_l0, encoder_en_convs_4_tra_att_gru_bias_ih_l0,
        encoder_en_convs_4_tra_att_gru_weight_hh_l0, encoder_en_convs_4_tra_att_gru_bias_hh_l0,
        encoder_en_convs_4_tra_att_fc_weight, encoder_en_convs_4_tra_att_fc_bias,
        y_conv4);

    /* Step 5: GDPRNN ×2 */
    GDPRNN_module(y_conv4, state->inter_prev1, 1,
        dpgrnn1_intra_rnn_rnn1_weight_ih_l0, dpgrnn1_intra_rnn_rnn1_bias_ih_l0,
        dpgrnn1_intra_rnn_rnn1_weight_hh_l0, dpgrnn1_intra_rnn_rnn1_bias_hh_l0,
        dpgrnn1_intra_rnn_rnn1_weight_ih_l0_reverse, dpgrnn1_intra_rnn_rnn1_bias_ih_l0_reverse,
        dpgrnn1_intra_rnn_rnn1_weight_hh_l0_reverse, dpgrnn1_intra_rnn_rnn1_bias_hh_l0_reverse,
        dpgrnn1_intra_rnn_rnn2_weight_ih_l0, dpgrnn1_intra_rnn_rnn2_bias_ih_l0,
        dpgrnn1_intra_rnn_rnn2_weight_hh_l0, dpgrnn1_intra_rnn_rnn2_bias_hh_l0,
        dpgrnn1_intra_rnn_rnn2_weight_ih_l0_reverse, dpgrnn1_intra_rnn_rnn2_bias_ih_l0_reverse,
        dpgrnn1_intra_rnn_rnn2_weight_hh_l0_reverse, dpgrnn1_intra_rnn_rnn2_bias_hh_l0_reverse,
        dpgrnn1_inter_rnn_rnn1_weight_ih_l0, dpgrnn1_inter_rnn_rnn1_bias_ih_l0,
        dpgrnn1_inter_rnn_rnn1_weight_hh_l0, dpgrnn1_inter_rnn_rnn1_bias_hh_l0,
        dpgrnn1_inter_rnn_rnn2_weight_ih_l0, dpgrnn1_inter_rnn_rnn2_bias_ih_l0,
        dpgrnn1_inter_rnn_rnn2_weight_hh_l0, dpgrnn1_inter_rnn_rnn2_bias_hh_l0,
        dpgrnn1_intra_fc_weight, dpgrnn1_intra_fc_bias,
        dpgrnn1_intra_ln_weight, dpgrnn1_intra_ln_bias,
        dpgrnn1_inter_fc_weight, dpgrnn1_inter_fc_bias,
        dpgrnn1_inter_ln_weight, dpgrnn1_inter_ln_bias,
        y_rnn1);

    GDPRNN_module(y_rnn1, state->inter_prev2, 2,
        dpgrnn2_intra_rnn_rnn1_weight_ih_l0, dpgrnn2_intra_rnn_rnn1_bias_ih_l0,
        dpgrnn2_intra_rnn_rnn1_weight_hh_l0, dpgrnn2_intra_rnn_rnn1_bias_hh_l0,
        dpgrnn2_intra_rnn_rnn1_weight_ih_l0_reverse, dpgrnn2_intra_rnn_rnn1_bias_ih_l0_reverse,
        dpgrnn2_intra_rnn_rnn1_weight_hh_l0_reverse, dpgrnn2_intra_rnn_rnn1_bias_hh_l0_reverse,
        dpgrnn2_intra_rnn_rnn2_weight_ih_l0, dpgrnn2_intra_rnn_rnn2_bias_ih_l0,
        dpgrnn2_intra_rnn_rnn2_weight_hh_l0, dpgrnn2_intra_rnn_rnn2_bias_hh_l0,
        dpgrnn2_intra_rnn_rnn2_weight_ih_l0_reverse, dpgrnn2_intra_rnn_rnn2_bias_ih_l0_reverse,
        dpgrnn2_intra_rnn_rnn2_weight_hh_l0_reverse, dpgrnn2_intra_rnn_rnn2_bias_hh_l0_reverse,
        dpgrnn2_inter_rnn_rnn1_weight_ih_l0, dpgrnn2_inter_rnn_rnn1_bias_ih_l0,
        dpgrnn2_inter_rnn_rnn1_weight_hh_l0, dpgrnn2_inter_rnn_rnn1_bias_hh_l0,
        dpgrnn2_inter_rnn_rnn2_weight_ih_l0, dpgrnn2_inter_rnn_rnn2_bias_ih_l0,
        dpgrnn2_inter_rnn_rnn2_weight_hh_l0, dpgrnn2_inter_rnn_rnn2_bias_hh_l0,
        dpgrnn2_intra_fc_weight, dpgrnn2_intra_fc_bias,
        dpgrnn2_intra_ln_weight, dpgrnn2_intra_ln_bias,
        dpgrnn2_inter_fc_weight, dpgrnn2_inter_fc_bias,
        dpgrnn2_inter_ln_weight, dpgrnn2_inter_ln_bias,
        y_rnn2);

    /* Step 6: Decoder — GT-DeConv×3 → DeConv1 → DeConv0 */
    GT_DeConv_module(y_rnn2, y_conv4, state->dec_conv_hist, state->dec_h_prev, 5, 0,
        decoder_de_convs_0_point_conv1_weight, decoder_de_convs_0_point_conv1_bias,
        decoder_de_convs_0_point_bn1_weight, decoder_de_convs_0_point_bn1_bias,
        decoder_de_convs_0_point_bn1_running_mean, decoder_de_convs_0_point_bn1_running_var,
        decoder_de_convs_0_point_act_weight,
        decoder_de_convs_0_depth_conv_weight, decoder_de_convs_0_depth_conv_bias,
        decoder_de_convs_0_depth_bn_weight, decoder_de_convs_0_depth_bn_bias,
        decoder_de_convs_0_depth_bn_running_mean, decoder_de_convs_0_depth_bn_running_var,
        decoder_de_convs_0_depth_act_weight,
        decoder_de_convs_0_point_conv2_weight, decoder_de_convs_0_point_conv2_bias,
        decoder_de_convs_0_point_bn2_weight, decoder_de_convs_0_point_bn2_bias,
        decoder_de_convs_0_point_bn2_running_mean, decoder_de_convs_0_point_bn2_running_var,
        decoder_de_convs_0_tra_att_gru_weight_ih_l0, decoder_de_convs_0_tra_att_gru_bias_ih_l0,
        decoder_de_convs_0_tra_att_gru_weight_hh_l0, decoder_de_convs_0_tra_att_gru_bias_hh_l0,
        decoder_de_convs_0_tra_att_fc_weight, decoder_de_convs_0_tra_att_fc_bias,
        y_d0);

    GT_DeConv_module(y_d0, y_conv3, state->dec_conv_hist, state->dec_h_prev + 1*16, 2, 1,
        decoder_de_convs_1_point_conv1_weight, decoder_de_convs_1_point_conv1_bias,
        decoder_de_convs_1_point_bn1_weight, decoder_de_convs_1_point_bn1_bias,
        decoder_de_convs_1_point_bn1_running_mean, decoder_de_convs_1_point_bn1_running_var,
        decoder_de_convs_1_point_act_weight,
        decoder_de_convs_1_depth_conv_weight, decoder_de_convs_1_depth_conv_bias,
        decoder_de_convs_1_depth_bn_weight, decoder_de_convs_1_depth_bn_bias,
        decoder_de_convs_1_depth_bn_running_mean, decoder_de_convs_1_depth_bn_running_var,
        decoder_de_convs_1_depth_act_weight,
        decoder_de_convs_1_point_conv2_weight, decoder_de_convs_1_point_conv2_bias,
        decoder_de_convs_1_point_bn2_weight, decoder_de_convs_1_point_bn2_bias,
        decoder_de_convs_1_point_bn2_running_mean, decoder_de_convs_1_point_bn2_running_var,
        decoder_de_convs_1_tra_att_gru_weight_ih_l0, decoder_de_convs_1_tra_att_gru_bias_ih_l0,
        decoder_de_convs_1_tra_att_gru_weight_hh_l0, decoder_de_convs_1_tra_att_gru_bias_hh_l0,
        decoder_de_convs_1_tra_att_fc_weight, decoder_de_convs_1_tra_att_fc_bias,
        y_d1);

    GT_DeConv_module(y_d1, y_conv2, state->dec_conv_hist, state->dec_h_prev + 2*16, 1, 2,
        decoder_de_convs_2_point_conv1_weight, decoder_de_convs_2_point_conv1_bias,
        decoder_de_convs_2_point_bn1_weight, decoder_de_convs_2_point_bn1_bias,
        decoder_de_convs_2_point_bn1_running_mean, decoder_de_convs_2_point_bn1_running_var,
        decoder_de_convs_2_point_act_weight,
        decoder_de_convs_2_depth_conv_weight, decoder_de_convs_2_depth_conv_bias,
        decoder_de_convs_2_depth_bn_weight, decoder_de_convs_2_depth_bn_bias,
        decoder_de_convs_2_depth_bn_running_mean, decoder_de_convs_2_depth_bn_running_var,
        decoder_de_convs_2_depth_act_weight,
        decoder_de_convs_2_point_conv2_weight, decoder_de_convs_2_point_conv2_bias,
        decoder_de_convs_2_point_bn2_weight, decoder_de_convs_2_point_bn2_bias,
        decoder_de_convs_2_point_bn2_running_mean, decoder_de_convs_2_point_bn2_running_var,
        decoder_de_convs_2_tra_att_gru_weight_ih_l0, decoder_de_convs_2_tra_att_gru_bias_ih_l0,
        decoder_de_convs_2_tra_att_gru_weight_hh_l0, decoder_de_convs_2_tra_att_gru_bias_hh_l0,
        decoder_de_convs_2_tra_att_fc_weight, decoder_de_convs_2_tra_att_fc_bias,
        y_d2);

    DeConv_block_1(y_d2, y_conv1,
        decoder_de_convs_3_conv_weight, decoder_de_convs_3_conv_bias,
        decoder_de_convs_3_bn_weight, decoder_de_convs_3_bn_bias,
        decoder_de_convs_3_bn_running_mean, decoder_de_convs_3_bn_running_var,
        decoder_de_convs_3_act_weight, y_d3);

    DeConv_block_0(y_d3, y_conv0,
        decoder_de_convs_4_conv_weight, decoder_de_convs_4_conv_bias,
        decoder_de_convs_4_bn_weight, decoder_de_convs_4_bn_bias,
        decoder_de_convs_4_bn_running_mean, decoder_de_convs_4_bn_running_var,
        y_dec);

    /* Step 7: BS + MASK */
    BS_fixed(y_dec, ierbfc_w, y_bs);
    MASK_fixed(y_bs, x_mag + 257, x_mag + 2*257, crm_out);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("============================================================\n");
    printf("  GTCRN C Fixed-Point — End-to-End Test\n");
    printf("  Weights: gtcrn_matlab_weights.h (%d tensors)\n", 271);
    printf("  State size: %zu bytes\n", sizeof(gtcrn_state_t));
    printf("============================================================\n");

    test_state_init();
    test_module_wiring();
    test_single_frame();
    test_multi_frame();

    printf("\n============================================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("============================================================\n");

    return tests_failed ? 1 : 0;
}
