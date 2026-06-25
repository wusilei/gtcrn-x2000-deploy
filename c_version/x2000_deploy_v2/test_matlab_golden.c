/**
 * test_matlab_golden.c — C vs MATLAB golden layer-by-layer comparison
 *
 * Reads MATLAB's 01_mag.bin as input (the SAME data MATLAB used),
 * runs through all layers, saves outputs in MATLAB-compatible
 * column-major (F-order) format for direct binary comparison.
 *
 * Compile: gcc -std=c99 -Wall -O2 -o test_matlab_golden test_matlab_golden.c gtcrn_fp.c -lm
 */
#include "gtcrn_matlab_weights.h"
#include "gtcrn_fp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Column-major <-> Row-major conversion helpers                       */
/* ------------------------------------------------------------------ */

/* Convert (C, W) from column-major (MATLAB) to row-major (C internal) */
static void cm_to_rm(const int32_t *src, int32_t *dst, int C, int W) {
    for (int c = 0; c < C; c++)
        for (int w = 0; w < W; w++)
            dst[c * W + w] = src[w * C + c];
}

/* Convert (C, W) from row-major (C internal) to column-major (MATLAB) */
static void rm_to_cm(const int32_t *src, int32_t *dst, int C, int W) {
    for (int w = 0; w < W; w++)
        for (int c = 0; c < C; c++)
            dst[w * C + c] = src[c * W + w];
}

/* Save int32 data as binary */
static void save_bin(const char *name, const int32_t *data, int n) {
    char path[256]; snprintf(path, sizeof(path), "dump_matlab_cmp/%s", name);
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(data, sizeof(int32_t), n, f); fclose(f); }
}

/* Save int16 as int32 (sign-extended, matching MATLAB int32() cast) */
static void save_bin_s16(const char *name, const int16_t *data, int n) {
    char path[256]; snprintf(path, sizeof(path), "dump_matlab_cmp/%s", name);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)data[i];
        fwrite(&v, sizeof(int32_t), 1, f);
    }
    fclose(f);
}

int main() {
    system("mkdir -p dump_matlab_cmp");

    /* ================================================================
     * STEP 1: Load MATLAB's 01_mag.bin (column-major, 3×257 s32f20)
     * ================================================================ */
    int32_t x_mag_cm[3*257];
    FILE *f01 = fopen("/tmp/01_mag.bin", "rb");
    if (!f01) { perror("01_mag.bin"); return 1; }
    fread(x_mag_cm, sizeof(int32_t), 3*257, f01); fclose(f01);

    /* Convert to C row-major: (3, 257) with channel-major layout */
    int32_t x_mag[3*257];
    cm_to_rm(x_mag_cm, x_mag, 3, 257);

    gtcrn_state_t state;
    gtcrn_state_init(&state);

    /* ================================================================
     * Layer 02: BM → output (3, 129) s32f20
     * ================================================================ */
    int32_t x_bm_rm[3*129];
    BM_fixed(x_mag, erb_erb_fc_weight, x_bm_rm);
    {
        int32_t tmp[3*129]; rm_to_cm(x_bm_rm, tmp, 3, 129);
        save_bin("02_bm.bin", tmp, 3*129);
    }

    /* ================================================================
     * Layer 03: SFE → output (9, 129) s32f20
     * ================================================================ */
    int32_t x_sfe_rm[9*129];
    SFE_fixed(x_bm_rm, 3, 129, x_sfe_rm);
    {
        int32_t tmp[9*129]; rm_to_cm(x_sfe_rm, tmp, 9, 129);
        save_bin("03_sfe.bin", tmp, 9*129);
    }

    /* ================================================================
     * Layer 04: Conv0 → output (16, 65) s32f20
     * ================================================================ */
    int32_t y_conv0_rm[16*65];
    Conv_block_0(x_sfe_rm,
        encoder_en_convs_0_conv_weight, encoder_en_convs_0_conv_bias,
        encoder_en_convs_0_bn_weight, encoder_en_convs_0_bn_bias,
        encoder_en_convs_0_bn_running_mean, encoder_en_convs_0_bn_running_var,
        encoder_en_convs_0_act_weight, y_conv0_rm);
    {
        int32_t tmp[16*65]; rm_to_cm(y_conv0_rm, tmp, 16, 65);
        save_bin("04_conv0.bin", tmp, 16*65);
    }

    /* ================================================================
     * Layer 05: Conv1 → output (16, 33) s32f20
     * ================================================================ */
    int32_t y_conv1_rm[16*33];
    Conv_block_1(y_conv0_rm,
        encoder_en_convs_1_conv_weight, encoder_en_convs_1_conv_bias,
        encoder_en_convs_1_bn_weight, encoder_en_convs_1_bn_bias,
        encoder_en_convs_1_bn_running_mean, encoder_en_convs_1_bn_running_var,
        encoder_en_convs_1_act_weight, y_conv1_rm);
    {
        int32_t tmp[16*33]; rm_to_cm(y_conv1_rm, tmp, 16, 33);
        save_bin("05_conv1.bin", tmp, 16*33);
    }

    /* ================================================================
     * Layers 06-08: GT_Conv (Encoder) ×3
     * ================================================================ */
    /* GT_Conv0: dil=1 */
    int32_t y_conv2_rm[16*33];
    GT_Conv_module(y_conv1_rm, state.enc_conv_hist, state.enc_h_prev, 1, 0,
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
        encoder_en_convs_2_tra_att_fc_weight, encoder_en_convs_2_tra_att_fc_bias, y_conv2_rm);
    { int32_t tmp[16*33]; rm_to_cm(y_conv2_rm, tmp, 16, 33); save_bin("06_gtconv0.bin", tmp, 16*33); }

    /* GT_Conv1: dil=2 */
    int32_t y_conv3_rm[16*33];
    GT_Conv_module(y_conv2_rm, state.enc_conv_hist, state.enc_h_prev + 1*16, 2, 1,
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
        encoder_en_convs_3_tra_att_fc_weight, encoder_en_convs_3_tra_att_fc_bias, y_conv3_rm);
    { int32_t tmp[16*33]; rm_to_cm(y_conv3_rm, tmp, 16, 33); save_bin("07_gtconv1.bin", tmp, 16*33); }

    /* GT_Conv2: dil=5 */
    int32_t y_conv4_rm[16*33];
    GT_Conv_module(y_conv3_rm, state.enc_conv_hist, state.enc_h_prev + 2*16, 5, 2,
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
        encoder_en_convs_4_tra_att_fc_weight, encoder_en_convs_4_tra_att_fc_bias, y_conv4_rm);
    { int32_t tmp[16*33]; rm_to_cm(y_conv4_rm, tmp, 16, 33); save_bin("08_gtconv2.bin", tmp, 16*33); }

    /* ================================================================
     * Layers 09-12: DPRNN (GDPRNN ×2)
     * ================================================================ */
    int32_t y_rnn1_rm[16*33];
    GDPRNN_module(y_conv4_rm, state.inter_prev1, 1,
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
        dpgrnn1_inter_ln_weight, dpgrnn1_inter_ln_bias, y_rnn1_rm);
    { int32_t tmp[16*33]; rm_to_cm(y_rnn1_rm, tmp, 16, 33); save_bin("10_interrnn1.bin", tmp, 16*33); }

    int32_t y_rnn2_rm[16*33];
    GDPRNN_module(y_rnn1_rm, state.inter_prev2, 2,
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
        dpgrnn2_inter_ln_weight, dpgrnn2_inter_ln_bias, y_rnn2_rm);
    { int32_t tmp[16*33]; rm_to_cm(y_rnn2_rm, tmp, 16, 33); save_bin("12_interrnn2.bin", tmp, 16*33); }

    /* ================================================================
     * Layers 13-15: GT_DeConv (Decoder) ×3
     * ================================================================ */
    /* GTD0: dil=5 */
    int32_t y_d0_rm[16*33];
    GT_DeConv_module(y_rnn2_rm, y_conv4_rm, state.dec_conv_hist, state.dec_h_prev, 5, 0,
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
        decoder_de_convs_0_tra_att_fc_weight, decoder_de_convs_0_tra_att_fc_bias, y_d0_rm);
    { int32_t tmp[16*33]; rm_to_cm(y_d0_rm, tmp, 16, 33); save_bin("13_gtd0.bin", tmp, 16*33); }

    /* GTD1: dil=2 */
    int32_t y_d1_rm[16*33];
    GT_DeConv_module(y_d0_rm, y_conv3_rm, state.dec_conv_hist, state.dec_h_prev + 1*16, 2, 1,
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
        decoder_de_convs_1_tra_att_fc_weight, decoder_de_convs_1_tra_att_fc_bias, y_d1_rm);
    { int32_t tmp[16*33]; rm_to_cm(y_d1_rm, tmp, 16, 33); save_bin("14_gtd1.bin", tmp, 16*33); }

    /* GTD2: dil=1 */
    int32_t y_d2_rm[16*33];
    GT_DeConv_module(y_d1_rm, y_conv2_rm, state.dec_conv_hist, state.dec_h_prev + 2*16, 1, 2,
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
        decoder_de_convs_2_tra_att_fc_weight, decoder_de_convs_2_tra_att_fc_bias, y_d2_rm);
    { int32_t tmp[16*33]; rm_to_cm(y_d2_rm, tmp, 16, 33); save_bin("15_gtd2.bin", tmp, 16*33); }

    /* ================================================================
     * Layer 16: DeConv1 → output (16, 65) s32f20
     * ================================================================ */
    int32_t y_d3_rm[16*65];
    DeConv_block_1(y_d2_rm, y_conv1_rm,
        decoder_de_convs_3_conv_weight, decoder_de_convs_3_conv_bias,
        decoder_de_convs_3_bn_weight, decoder_de_convs_3_bn_bias,
        decoder_de_convs_3_bn_running_mean, decoder_de_convs_3_bn_running_var,
        decoder_de_convs_3_act_weight, y_d3_rm);
    { int32_t tmp[16*65]; rm_to_cm(y_d3_rm, tmp, 16, 65); save_bin("16_deconv1.bin", tmp, 16*65); }

    /* ================================================================
     * Layer 17: DeConv0 → output (2, 129) s16f15
     * ================================================================ */
    int16_t y_dec_s16[2*129];
    DeConv_block_0(y_d3_rm, y_conv0_rm,
        decoder_de_convs_4_conv_weight, decoder_de_convs_4_conv_bias,
        decoder_de_convs_4_bn_weight, decoder_de_convs_4_bn_bias,
        decoder_de_convs_4_bn_running_mean, decoder_de_convs_4_bn_running_var, y_dec_s16);
    /* Convert s16f15 to s32 and save in column-major */
    {
        int32_t tmp_rm[2*129], tmp_cm[2*129];
        for (int i = 0; i < 2*129; i++) tmp_rm[i] = (int32_t)y_dec_s16[i];
        rm_to_cm(tmp_rm, tmp_cm, 2, 129);
        save_bin("17_deconv0.bin", tmp_cm, 2*129);
    }

    /* ================================================================
     * Layer 18: BS → output (2, 257) s16f15
     * ================================================================ */
    int16_t y_bs_s16[2*257];
    BS_fixed(y_dec_s16, erb_ierb_fc_weight, y_bs_s16);
    {
        int32_t tmp_rm[2*257], tmp_cm[2*257];
        for (int i = 0; i < 2*257; i++) tmp_rm[i] = (int32_t)y_bs_s16[i];
        rm_to_cm(tmp_rm, tmp_cm, 2, 257);
        save_bin("18_bs.bin", tmp_cm, 2*257);
    }

    /* ================================================================
     * Layer 19: CRM → output (2, 257) s32f20
     * ================================================================ */
    int32_t y_crm_rm[2*257];
    /* MASK_fixed takes mask(s16f15), real_in(s32f20), imag_in(s32f20) */
    MASK_fixed(y_bs_s16, x_mag + 257, x_mag + 2*257, y_crm_rm);
    {
        int32_t tmp[2*257]; rm_to_cm(y_crm_rm, tmp, 2, 257);
        save_bin("19_crm.bin", tmp, 2*257);
    }

    printf("All 18 layers dumped to dump_matlab_cmp/ (column-major, matching MATLAB)\n");
    printf("Ready for direct binary comparison with MATLAB golden.\n");
    return 0;
}
