/**
 * agc.c — 语音自动增益控制 (AGC) 模块 实现
 * ==========================================
 * 定点运算, 无浮点依赖, 静态状态 (无需堆分配)
 */
#include "agc.h"
#include <stdint.h>
#include <string.h>

static int32_t g_energy_smooth = 0;
static int     g_first_frame   = 1;
static int32_t g_gain_high_num = AGC_GAIN_HIGH_NUM;
static int32_t g_gain_low_num  = AGC_GAIN_LOW_NUM;

static inline int16_t s16_clamp(int32_t v) {
    if (v >  32767) return  32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

void agc_init(void) {
    g_energy_smooth = 0;
    g_first_frame   = 1;
}

int energy_calc_frame(const int16_t *buf, int len) {
    if (!buf || len <= 0) return 0;
    int64_t sq = 0;
    for (int i = 0; i < len; i++)
        sq += (int64_t)buf[i] * buf[i];
    return (int)(sq / len);
}

void agc_process(int16_t *pcm, int len, int energy_current) {
    if (!pcm || len <= 0) return;

    /* 1. 能量平滑 */
    int energy_cur;
    if (energy_current > 0) {
        energy_cur = energy_current;
        g_energy_smooth = (int32_t)(((int64_t)AGC_SMOOTH_COEFF * g_energy_smooth
                           + (int64_t)(32768 - AGC_SMOOTH_COEFF) * energy_cur) >> 15);
    } else {
        energy_cur = energy_calc_frame(pcm, len);
        g_energy_smooth = (int32_t)(((int64_t)AGC_SMOOTH_COEFF * g_energy_smooth
                           + (int64_t)(32768 - AGC_SMOOTH_COEFF) * energy_cur) >> 15);
    }

    /* 2. 分级增益决策 (基于原始能量) */
    int gain_num = (energy_cur > AGC_ENERGY_THR ||
                    g_energy_smooth > AGC_SMOOTH_THR)
                   ? g_gain_high_num : g_gain_low_num;
    /* 首帧跳过增益, 积累能量估计 */
    if (g_first_frame) {
        g_first_frame = 0;
        return;
    }

    /* 3. 增益放大 + 钳位 */
    for (int i = 0; i < len; i++) {
        int32_t v = (int32_t)pcm[i] * gain_num / AGC_GAIN_DEN;
        pcm[i] = s16_clamp(v);
    }
}
