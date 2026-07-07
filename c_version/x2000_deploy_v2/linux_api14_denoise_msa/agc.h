/**
 * agc.h — 语音 AGC + 降噪联合模块 (v20 — linux_api14)
 * ======================================================
 *
 * 数据流 (3 阶段):
 *   1. MIC → energy_calculate_and_smooth_s16 → current_energy + *energy_out
 *   2. 二分阈值决策: current_energy / energy_out → goal_val
 *   3. MIC → voice_NoiseReductionAndAGC(in, out, goal, state_flag)
 *
 * voice_NoiseReductionAndAGC 内部 (AGC先, 降噪后 — 前置AGC):
 *   state_flag=0: bypass
 *   state_flag=1: 只降噪 ( DENOISE )
 *   state_flag=2: 只AGC (voice_AGC LMS + 固定增益)
 *   state_flag=3: AGC + 降噪 (前置AGC管线)
 */
#ifndef AGC_H
#define AGC_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STATE_FLAG_NONE     0
#define STATE_FLAG_DENOISE  1
#define STATE_FLAG_AGC      2
#define STATE_FLAG_BOTH     3

#define ENERGY_ABS_THR_HIGH   250
#define ENERGY_ABS_THR_LOW    144
#define ENERGY_SMOOTH_THR     144
#define ENERGY_HISTORY_FAC    20783

#define AGC_GOAL_HIGH  (1LL << 20)
#define AGC_GOAL_LOW   (1LL << 10)

#define AGC_GAIN_HIGH_NUM 5
#define AGC_GAIN_LOW_NUM  4
#define AGC_GAIN_DEN      2

#define AGC_V_A        800
#define AGC_V_MU1      100
#define AGC_V_MU2      800
#define AGC_V_MU3      1600
#define AGC_V_SHIFT1   10
#define AGC_V_SHIFT2   45
#define AGC_V_SHIFT3   15
#define AGC_V_GAIN_MIN 31
#define AGC_V_GAIN_MAX 32767

unsigned short energy_calculate_and_smooth_s16(short *data_in,
    unsigned short data_len, unsigned short history_fac,
    unsigned short *energy_out);

void voice_NoiseReductionAndAGC(short *in, short *out,
    long long goal, short flag);

void agc_init(void);

#ifdef __cplusplus
}
#endif
#endif
