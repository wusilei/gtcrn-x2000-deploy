/**
 * GTCRN 纯 C 定点降噪 — DSP 算法 API
 * =====================================
 * 静态链接: libgtcrn.a + libm
 *
 * 帧规格: 200 样本/帧 @ 8kHz (25ms), 单声道 int16
 * 延迟:   ~50ms (上采样 + STFT + 推理 + ISTFT + 下采样)
 * 内存:   ~70KB 状态 + ~302KB 权重 + ~50KB 栈 = ~420KB
 *
 * 替换: libdenoise_v8_agc (NNoM) → libgtcrn (纯 C 定点)
 */

#ifndef NOISE_REDUCTION_H
#define NOISE_REDUCTION_H

#ifdef __cplusplus
extern "C" {
#endif

/** 初始化降噪器 (启动时调用一次, ~0.5ms) */
void noise_init(void);

/**
 * 降噪处理 (每 25ms 调用一次)
 * @param voiceIn   输入 PCM, 200 个 int16 @ 8kHz
 * @param voiceOut  输出 PCM, 200 个 int16 @ 8kHz (增强后)
 *
 * @note 帧长从旧库的 100 样本/12.5ms 改为 200 样本/25ms
 *       以匹配 GTCRN 推理耗时 (28ms @ X2000 MIPS)
 */
void noise_reduction(short *voiceIn, short *voiceOut);

/** 销毁降噪器 (关机时调用, 可选) */
void noise_deinit(void);

#ifdef __cplusplus
}
#endif
#endif
