/**
 * agc.c — 语音 AGC + 降噪联合模块 实现 (v21 — linux_api15, 混合AGC)
 * ==================================================================
 * voice_NoiseReductionAndAGC: state_flag=3 → 混合AGC: 前置轻量固定增益 → 降噪 → 后置主AGC (voice_AGC LMS + 固定增益) 
 */
#include "agc.h"
#include "noise_reduction.h"
#include <stdint.h>
#include <string.h>

static long long g_v_Px_pre = 0, g_v_Py_pre = 0, g_v_g_pre = 0;
static int g_v_inited = 0, g_first_frame = 1;

static inline int16_t s16_clamp(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

static void voice_agc_core(short *data_in, short *data_out,
                           unsigned int data_len, unsigned int first_flag,
                           long long goal_val)
{
    long long Px_cur, Py_cur, g_cur;
    unsigned int i; short x, y, mu;

    if (first_flag == 0) {
        x = data_in[0]; data_out[0] = x;
        g_v_Px_pre = g_v_Py_pre = (long long)x * x;
        g_v_g_pre = 32768; g_v_inited = 1; i = 1;
    } else {
        if (!g_v_inited) {
            if (data_len > 0) {
                x = data_in[0]; data_out[0] = x;
                g_v_Px_pre = g_v_Py_pre = (long long)x * x;
                g_v_g_pre = 32768; g_v_inited = 1;
            } i = 1;
        } else i = 0;
    }

    for (; i < data_len; i++) {
        x = data_in[i];
        if (x > 1024 || x < -1024) mu = AGC_V_MU1;
        else if (x > -16 && x < 16) mu = AGC_V_MU3;
        else mu = AGC_V_MU2;

        Px_cur = (AGC_V_A * g_v_Px_pre + ((1LL << AGC_V_SHIFT1) - AGC_V_A)
                  * (long long)((int)x * (int)x)) >> AGC_V_SHIFT1;
        g_cur = g_v_g_pre * ((1LL << (60 - AGC_V_SHIFT2))
               + ((mu * Px_cur * (goal_val - g_v_Py_pre)) >> AGC_V_SHIFT2));
        y = (short)((g_cur * x) >> (AGC_V_SHIFT3 + (60 - AGC_V_SHIFT2)));
        g_cur >>= (60 - AGC_V_SHIFT2);
        if (g_cur < AGC_V_GAIN_MIN) g_cur = AGC_V_GAIN_MIN;
        if (g_cur > AGC_V_GAIN_MAX) g_cur = AGC_V_GAIN_MAX;
        Py_cur = (g_cur * g_cur * Px_cur) >> (AGC_V_SHIFT3 * 2);
        g_v_Px_pre = Px_cur; g_v_Py_pre = Py_cur; g_v_g_pre = g_cur;
        data_out[i] = y;
    }
}

void agc_init(void) {
    g_v_Px_pre = g_v_Py_pre = g_v_g_pre = 0;
    g_v_inited = 0; g_first_frame = 1;
    noise_init();
}

unsigned short energy_calculate_and_smooth_s16(short *data_in,
    unsigned short data_len, unsigned short history_fac,
    unsigned short *energy_out)
{
    unsigned int data_sum = 0;
    for (unsigned int i = 0; i < data_len; i++) {
        short v = data_in[i];
        data_sum += (v < 0) ? (unsigned short)(-v) : (unsigned short)v;
    }
    unsigned short cur = (unsigned short)(data_sum / data_len);
    *energy_out = (unsigned short)((((unsigned int)history_fac * (*energy_out))
                 + ((0x7fffU - history_fac) * cur) + 16384) >> 15);
    return cur;
}

void voice_NoiseReductionAndAGC(short *in, short *out, long long goal, short flag)
{
    if (!in || !out) return;

    switch (flag) {
    case STATE_FLAG_NONE:
        if (in != out) memcpy(out, in, 200 * sizeof(short));
        break;

    case STATE_FLAG_DENOISE:
        noise_reduction(in, out);
        break;

    case STATE_FLAG_AGC: {
        short local[200]; memcpy(local, in, sizeof(local));
        int vf = g_v_inited ? 1 : 0;
        voice_agc_core(local, local, 200, (unsigned int)vf, goal);
        if (!g_first_frame) {
            int gn = (goal == AGC_GOAL_HIGH) ? AGC_GAIN_HIGH_NUM : AGC_GAIN_LOW_NUM;
            for (int i = 0; i < 200; i++) {
                int32_t v = (int32_t)local[i] * gn / AGC_GAIN_DEN;
                out[i] = s16_clamp(v);
            }
        } else memcpy(out, local, sizeof(local));
        break;
    }

    case STATE_FLAG_BOTH: {
        /* 混合AGC: 前置轻量固定增益 → 降噪 → 后置主AGC (voice_AGC LMS + 固定增益) */
        #define PRE_GAIN_NUM 3   /* 前置增益 3/2=1.5x, 轻量防切削 */
        #define PRE_GAIN_DEN 2

        /* Stage 1: 前置轻量固定增益 (不做LMS, 避免状态污染) */
        short pre_out[200];
        for (int i = 0; i < 200; i++) {
            int32_t v = (int32_t)in[i] * PRE_GAIN_NUM / PRE_GAIN_DEN;
            pre_out[i] = s16_clamp(v);
        }

        /* Stage 2: GTCRN 降噪 */
        short nr_out[200];
        noise_reduction(pre_out, nr_out);

        /* Stage 3: 后置主 AGC (voice_AGC LMS + 分级固定增益) */
        int vf = g_v_inited ? 1 : 0;
        voice_agc_core(nr_out, nr_out, 200, (unsigned int)vf, goal);

        if (!g_first_frame) {
            int gn = (goal == AGC_GOAL_HIGH) ? AGC_GAIN_HIGH_NUM : AGC_GAIN_LOW_NUM;
            for (int i = 0; i < 200; i++) {
                int32_t v = (int32_t)nr_out[i] * gn / AGC_GAIN_DEN;
                out[i] = s16_clamp(v);
            }
        } else {
            memcpy(out, nr_out, sizeof(nr_out));
        }
        break;
    }

    default:
        if (in != out) memcpy(out, in, 200 * sizeof(short));
        break;
    }
    if (g_first_frame) g_first_frame = 0;
}
