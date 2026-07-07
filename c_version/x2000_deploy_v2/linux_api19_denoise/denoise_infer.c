/* denoise_inferc — Full DENOISE inference (strong symbol, overrides weak stub) */
#include "denoise_matlab_weights.h"
#include "denoise_fp.h"

void denoise_infer_frame(const float *real_in, const float *imag_in,
                       denoise_state_t *state,
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

    mag_gen(real_in, imag_in, 257, x_mag);
    BM_fixed(x_mag, erbfc_w, x_bm);
    SFE_fixed(x_bm, 3, 129, x_sfe);

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
        encoder_en_convs_2_tra_att_fc_weight, encoder_en_convs_2_tra_att_fc_bias, y_conv2);
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
        encoder_en_convs_3_tra_att_fc_weight, encoder_en_convs_3_tra_att_fc_bias, y_conv3);
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
        encoder_en_convs_4_tra_att_fc_weight, encoder_en_convs_4_tra_att_fc_bias, y_conv4);

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
        dpgrnn1_inter_ln_weight, dpgrnn1_inter_ln_bias, y_rnn1);
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
        dpgrnn2_inter_ln_weight, dpgrnn2_inter_ln_bias, y_rnn2);

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
        decoder_de_convs_0_tra_att_fc_weight, decoder_de_convs_0_tra_att_fc_bias, y_d0);
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
        decoder_de_convs_1_tra_att_fc_weight, decoder_de_convs_1_tra_att_fc_bias, y_d1);
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
        decoder_de_convs_2_tra_att_fc_weight, decoder_de_convs_2_tra_att_fc_bias, y_d2);
    DeConv_block_1(y_d2, y_conv1,
        decoder_de_convs_3_conv_weight, decoder_de_convs_3_conv_bias,
        decoder_de_convs_3_bn_weight, decoder_de_convs_3_bn_bias,
        decoder_de_convs_3_bn_running_mean, decoder_de_convs_3_bn_running_var,
        decoder_de_convs_3_act_weight, y_d3);
    DeConv_block_0(y_d3, y_conv0,
        decoder_de_convs_4_conv_weight, decoder_de_convs_4_conv_bias,
        decoder_de_convs_4_bn_weight, decoder_de_convs_4_bn_bias,
        decoder_de_convs_4_bn_running_mean, decoder_de_convs_4_bn_running_var, y_dec);
    BS_fixed(y_dec, ierbfc_w, y_bs);
    MASK_fixed(y_bs, x_mag + 257, x_mag + 2*257, crm_out);
}

/* Q15 integer version — real_in/imag_in in Q15 (int32_t).
 * Calls mag_gen_q15 instead of mag_gen; body otherwise identical. */
void denoise_infer_frame_q15(const int32_t *real_in, const int32_t *imag_in,
                           denoise_state_t *state,
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

    mag_gen_q15(real_in, imag_in, 257, x_mag);
    BM_fixed(x_mag, erbfc_w, x_bm);
    SFE_fixed(x_bm, 3, 129, x_sfe);

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
        encoder_en_convs_2_tra_att_fc_weight, encoder_en_convs_2_tra_att_fc_bias, y_conv2);
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
        encoder_en_convs_3_tra_att_fc_weight, encoder_en_convs_3_tra_att_fc_bias, y_conv3);
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
        encoder_en_convs_4_tra_att_fc_weight, encoder_en_convs_4_tra_att_fc_bias, y_conv4);

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
        dpgrnn1_inter_ln_weight, dpgrnn1_inter_ln_bias, y_rnn1);
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
        dpgrnn2_inter_ln_weight, dpgrnn2_inter_ln_bias, y_rnn2);

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
        decoder_de_convs_0_tra_att_fc_weight, decoder_de_convs_0_tra_att_fc_bias, y_d0);
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
        decoder_de_convs_1_tra_att_fc_weight, decoder_de_convs_1_tra_att_fc_bias, y_d1);
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
        decoder_de_convs_2_tra_att_fc_weight, decoder_de_convs_2_tra_att_fc_bias, y_d2);
    DeConv_block_1(y_d2, y_conv1,
        decoder_de_convs_3_conv_weight, decoder_de_convs_3_conv_bias,
        decoder_de_convs_3_bn_weight, decoder_de_convs_3_bn_bias,
        decoder_de_convs_3_bn_running_mean, decoder_de_convs_3_bn_running_var,
        decoder_de_convs_3_act_weight, y_d3);
    DeConv_block_0(y_d3, y_conv0,
        decoder_de_convs_4_conv_weight, decoder_de_convs_4_conv_bias,
        decoder_de_convs_4_bn_weight, decoder_de_convs_4_bn_bias,
        decoder_de_convs_4_bn_running_mean, decoder_de_convs_4_bn_running_var, y_dec);
    BS_fixed(y_dec, ierbfc_w, y_bs);
    MASK_fixed(y_bs, x_mag + 257, x_mag + 2*257, crm_out);
}
