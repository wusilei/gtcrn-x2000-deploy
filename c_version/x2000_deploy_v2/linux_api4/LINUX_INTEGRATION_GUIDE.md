# GTCRN 纯 C 定点 — Linux 侧降噪集成指南 (v11 final, 50ms iccom)

## 概述

纯 C 定点 GTCRN 降噪引擎，运行于 X2000 Linux 侧 (XBurst@II.V2, MIPS32R2, 无 FPU)。
通过 iccom rf11/wf12 与 RTOS/DSP 侧交换 8kHz PCM 音频。

**版本**: v11 final (2026-06-23)
**状态**: ✅ X2000 验证通过, 全部 bit-exact with PC
**与 v10 (linux_api3) 差异**: iccom 帧规格 200→400 (25ms→50ms), 其余完全一致

---

## API

```c
// noise_reduction.h — API 不变
void noise_init(void);                              // 启动时调用一次
void noise_reduction(short *in, short *out);         // 每 25ms 调用一次 (200×int16)
void noise_deinit(void);                             // 关机时调用
```

> `noise_reduction` 接口保持 200×int16 @ 8kHz / 25ms。iccom 模式内部每 50ms 调用两次。

---

## iccom 数据流 (50ms)

```
RTOS (DSP 侧)                     Linux 侧
─────────────                     ────────
每 50ms: write(rf11, 400×int16) → read(rf11, 400)
                                    → noise_reduction(in[0..199],   out[0..199])
                                    → noise_reduction(in[200..399], out[200..399])
                                 ← write(wf12, 400×int16)
每 50ms: read(wf12, 400)
```

**通道**: `rf11` (RTOS→Linux) / `wf12` (Linux→RTOS)，400×int16 @ 8kHz，50ms/帧

---

## 流水线 (内部不变)

```
noise_reduction(in, out): 每25ms调用 (iccom 模式下每50ms调用2次)
  1. 200×int16@8kHz → ÷32768 → 上采样 → 400 float @ 16kHz → FIFO[2048]
  2. while g_fifo_count ≥ 512 (WIN_LEN):
       a. 取最旧 512 样本 (时间正序, read_wpos 逐帧推进)
       b. × sqrt-Hann窗 → kiss_fftr → STFT (257 bins)
       c. GTCRN 推理 → CRM (2×257 s32f20)
       d. CRM ÷ 1048576 → kiss_fftri → ÷512 × sqrt-Hann → OLA (COLA自动)
       e. 输出 256 样本到 g_out_fifo
  3. if g_out_count ≥ 400: 下采样 → ×32768 → 200 int16 @ 8kHz output
```

---

## 关键参数

| 参数 | 值 |
|:--|:--|
| 内部帧规格 | 200 sample @ 8kHz (25ms) |
| iccom 帧规格 | 400 sample @ 8kHz (50ms) |
| STFT | 512-pt, 256 hop, sqrt-Hann 窗, 无 mirror padding |
| GTCRN 推理 | 纯 C 定点 (~28ms/帧 @ X2000) |
| 内存 | ~420KB (state 70KB + 权重 302KB + 缓冲 50KB) |
| 延迟 | ~75ms (50ms iccom + 25ms 内部缓冲) |

---

## 编译

### X2000 (MIPS32R2)
```bash
mips-linux-gnu-gcc -O3 -std=c99 -march=mips32r2 -msoft-float \
    -o gtcrn_linux_mips \
    noise_reduction.c gtcrn_fp.c gtcrn_infer.c \
    kiss_fft.c kiss_fftr.c gtcrn_linux.c -lm -static
```

### PC (x86, 测试用)
```bash
gcc -O3 -std=c99 -o gtcrn_linux \
    noise_reduction.c gtcrn_fp.c gtcrn_infer.c \
    kiss_fft.c kiss_fftr.c gtcrn_linux.c -lm
```

---

## 运行

```bash
# X2000 iccom 模式 (生产) — 400 sample/50ms
./gtcrn_linux

# PC 8kHz 测试 (对讲机实录等)
dd if=input_8k.wav bs=44 skip=1 | ./gtcrn_linux test > output_8k.pcm

# PC 16kHz 测试 (MATLAB 导出)
dd if=input_16k.wav bs=44 skip=1 | ./gtcrn_linux test16k > output_8k.pcm
```

---

## 文件清单

```
linux_api4/
├── gtcrn_linux.c              ← 主程序 (iccom 50ms / test / test16k)  [唯一改动]
├── noise_reduction.h          公开 API (与 linux_api3 相同)
├── noise_reduction.c          流式降噪适配层 (与 linux_api3 相同)
├── gtcrn_fp.h                 GTCRN 定点推理头文件 (与 linux_api3 相同)
├── gtcrn_fp.c                 GTCRN 定点推理实现 (与 linux_api3 相同)
├── gtcrn_infer.c              推理管线 (与 linux_api3 相同)
├── gtcrn_matlab_weights.h     定点权重 (与 linux_api3 相同)
├── kiss_fft.c/h               KissFFT 复数 FFT (与 linux_api3 相同)
├── kiss_fftr.c/h              KissFFT 实数 FFT (与 linux_api3 相同)
├── gtcrn_linux                PC 可执行文件
├── gtcrn_linux_mips           X2000 MIPS32R2 可执行文件 (880KB)
└── LINUX_INTEGRATION_GUIDE.md 本文件
```

---

## 验证结果

| 验证项 | 结果 |
|:--|:--|
| 5 文件 MATLAB 测试 | 全部 bit-exact (X2000 vs PC vs linux_api3) ✅ |
| 8 文件对讲机实录 | 全部 bit-exact (X2000 vs PC vs linux_api3) ✅ |
| test/test16k 模式 | 与 linux_api3 完全一致 ✅ |
| 滋滋噪声 | ✅ 已消除 |
| 人声可懂度 | ✅ 正常 |
| iccom 50ms 适配 | ✅ 仅改 gtcrn_linux.c 5 行 |

---

## v10→v11 唯一差异 (gtcrn_linux.c)

```diff
-            short in[FRAME_IN], out[FRAME_IN];
-            ssize_t nr = read(fd_rf, in, FRAME_IN * sizeof(short));
+            #define FRAME_50MS (FRAME_IN * 2)
+            short in[FRAME_50MS], out[FRAME_50MS];
+            ssize_t nr = read(fd_rf, in, FRAME_50MS * sizeof(short));
...
-            noise_reduction(in, out);
-            write(fd_wf, out, FRAME_IN * sizeof(short));
+            noise_reduction(in,           out);
+            noise_reduction(in + FRAME_IN, out + FRAME_IN);
+            write(fd_wf, out, FRAME_50MS * sizeof(short));
```
