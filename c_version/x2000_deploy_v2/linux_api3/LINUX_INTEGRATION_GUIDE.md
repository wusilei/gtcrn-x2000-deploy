# GTCRN 纯 C 定点 — Linux 侧降噪集成指南 (v10 final)

## 概述

纯 C 定点 GTCRN 降噪引擎，运行于 X2000 Linux 侧 (XBurst@II.V2, MIPS32R2, 无 FPU)。
通过 iccom rf11/wf12 与 RTOS/DSP 侧交换 8kHz PCM 音频。

**版本**: v10 final (2026-06-23)
**状态**: ✅ X2000 验证通过, 全部 bit-exact with PC

---

## 文件清单

```
linux_api3/
├── gtcrn_linux.c              主程序 (3 模式: iccom / test / test16k)
├── noise_reduction.h           公开 API
├── noise_reduction.c           流式降噪适配层 (sqrt-Hann, 时序正序, COLA)
├── gtcrn_fp.h                  GTCRN 定点推理头文件 (Q格式/state/F2Q20饱和)
├── gtcrn_fp.c                  GTCRN 定点推理实现 (纯C, 19层)
├── gtcrn_infer.c               推理管线 (strong symbol)
├── gtcrn_matlab_weights.h      定点权重 (271 tensors, ~302KB)
├── kiss_fft.c/h                KissFFT 复数 FFT (512-pt)
├── kiss_fftr.c/h               KissFFT 实数 FFT (前向+逆向)
├── gtcrn_linux                 PC 可执行文件
├── gtcrn_linux_mips            X2000 MIPS32R2 可执行文件 (880KB)
├── CHANGES.md                  完整修复记录
└── test_output/                测试输出
```

---

## API

```c
// noise_reduction.h
void noise_init(void);                              // 启动时调用一次
void noise_reduction(short *in, short *out);         // 每 25ms 调用一次
void noise_deinit(void);                             // 关机时调用
```

- `in`: 200 × int16 @ 8kHz (25ms 帧)
- `out`: 200 × int16 @ 8kHz (增强后)
- 状态: ~70KB static (gtcrn_state_t + FIFO + OLA buffer)
- 权重: ~302KB (gtcrn_matlab_weights.h, 编译时嵌入)
- 总内存: ~420KB

---

## 流水线

```
8kHz PCM (200 sample, int16)
  → ÷32768 → 上采样 → 400 float @ 16kHz → FIFO[2048]
  → while g_fifo_count ≥ 512:
      a. 取最旧 512 样本 (时间正序, read_wpos 逐帧推进)
      b. × sqrt-Hann窗 → kiss_fftr → STFT (257 bins)
      c. GTCRN 推理 (gtcrn_infer_frame) → CRM (2×257 s32f20)
      d. CRM ÷ 1048576 → kiss_fftri → ÷512 × sqrt-Hann → OLA
      e. OLA 输出 256 样本 → g_out_fifo (COLA 自动满足, 无 wsum)
  → if g_out_count ≥ 400: 下采样 → ×32768 → 200 int16 @ 8kHz output
```

### 关键参数

| 参数 | 值 |
|:--|:--|
| 帧规格 | 200 sample @ 8kHz (25ms) |
| STFT | 512-pt, 256 hop, sqrt-Hann 窗, 无 mirror padding |
| 分析/合成窗 | sqrt-Hann (sin(πi/511)), COLA 自动满足 |
| STFT 触发条件 | FIFO ≥ 512 有效样本 |
| STFT 帧处理序 | 最旧→最新 (时间正序) |
| GTCRN 推理 | 纯 C 定点 (~28ms/帧 @ X2000) |
| 内存 | ~420KB (state 70KB + 权重 302KB + 缓冲 50KB) |

---

## 编译

### PC (x86, 测试用)
```bash
gcc -O3 -std=c99 -o gtcrn_linux \
    noise_reduction.c gtcrn_fp.c gtcrn_infer.c \
    kiss_fft.c kiss_fftr.c gtcrn_linux.c -lm
```

### X2000 (MIPS32R2)
```bash
mips-linux-gnu-gcc -O3 -std=c99 -march=mips32r2 -msoft-float \
    -o gtcrn_linux_mips \
    noise_reduction.c gtcrn_fp.c gtcrn_infer.c \
    kiss_fft.c kiss_fftr.c gtcrn_linux.c -lm -static
```

---

## 运行

### iccom 模式 (生产)
```bash
./gtcrn_linux
# 从 /dev/iccom_rf11 读入, 写入 /dev/iccom_wf12
```

### test 模式 (8kHz 原生, 对讲机实录等)
```bash
dd if=input_8k.wav bs=44 skip=1 | ./gtcrn_linux test > output_8k.pcm
```

### test16k 模式 (16kHz MATLAB 导出)
```bash
dd if=input_16k.wav bs=44 skip=1 | ./gtcrn_linux test16k > output_8k.pcm
```

---

## iccom 数据流

```
RTOS (DSP 侧)                     Linux 侧
─────────────                     ────────
每 25ms: write(rf11, 200×int16) → read(rf11)
                                    → noise_reduction(in, out)
                                 ← write(wf12, 200×int16)
每 25ms: read(wf12)
```

通道: `rf11` (RTOS→Linux) / `wf12` (Linux→RTOS)，200×int16 @ 8kHz，25ms/帧。

---

## 验证结果

| 验证项 | 结果 |
|:--|:--|
| 5 文件 MATLAB 测试 | 全部 bit-exact (X2000 vs PC) |
| 8 文件对讲机实录 | 全部 bit-exact (X2000 vs PC) |
| 滋滋噪声 | ✅ 已消除 |
| 人声可懂度 | ✅ 正常 |
| 输出时长 | 与输入一致 (10s→10s, 42s→42s) |
| 实时性 | ⚠️ ~1.9× 实时 (47.5ms/帧, 需<25ms) |
| iccom 联调 | 待 DSP 工程师对接 |

---

## v9→v10 修复汇总

| # | Bug | 修复 |
|:--|:--|:--|
| 1 | 16kHz 测试音频被当 8kHz 处理 | test/test16k 双模式 |
| 2 | FIFO 不足 512 就做 512-pt STFT | ≥WIN_LEN(512) 门槛 |
| 3 | STFT 读指针不推进 | read_wpos 逐帧 +WIN_INC |
| 4 | STFT 帧时间反序 (GRU 破坏) | 最旧→最新 正序 |
| 5 | Hann² COLA + wsum 边缘噪声 | sqrt-Hann 自动 COLA |
| 6 | F2Q20 溢出回绕 | inline INT32 饱和 |
