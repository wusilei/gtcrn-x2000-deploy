/**
 * agc.h — 语音自动增益控制 (AGC) 模块
 * ======================================
 * 适配 GTCRN 管线: 200 samples @ 8kHz
 * 功能: 能量平滑 → 分级增益 → S16 钳位
 */
#ifndef AGC_H
#define AGC_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGC_SMOOTH_COEFF  20783   /* 能量平滑系数 (Q15) */
#define AGC_ENERGY_THR    200     /* 瞬时能量阈值 (Q0) */
#define AGC_SMOOTH_THR    45      /* 平滑能量阈值 (Q0) */
#define AGC_GAIN_HIGH_NUM 3       /* 远讲(低能量)增益分子 (3/2=1.5x) */
#define AGC_GAIN_LOW_NUM  2       /* 近讲(高能量)增益分子 (2/2=1.0x) */
#define AGC_GAIN_DEN      2       /* 增益分母 */

/** 初始化 AGC 状态 */
void agc_init(void);

/**
 * 处理一帧音频 (原地)
 * @param pcm             [in/out] int16 PCM
 * @param len             样本数
 * @param energy_current  降噪前原始帧平均能量 (Q0, >0 时使用)
 */
void agc_process(int16_t *pcm, int len, int energy_current);

/** 计算一帧音频平均能量 (非平滑, 供降噪前置能量采集) */
int energy_calc_frame(const int16_t *buf, int len);

#ifdef __cplusplus
}
#endif
#endif
