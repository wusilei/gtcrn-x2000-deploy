# GTCRN 纯 C 定点 — Linux 侧降噪集成指南

## 文件清单

```
linux_api/
├── noise_reduction.h           公开 API 头文件
├── gtcrn_fp.h                  GTCRN 定点推理头文件 (Q格式/state/原型)
├── gtcrn_fp.c                  GTCRN 定点推理实现 (纯 C, 8 Bug修复, float优化)
├── gtcrn_infer.c               GTCRN 完整推理管线 (strong symbol, 19层串联)
├── gtcrn_matlab_weights.h      定点权重 header (271 tensors, ~302KB)
└── LINUX_INTEGRATION_GUIDE.md  本文件
```

---

## 部署信息

| 属性 | 值 |
|------|-----|
| 部署位置 | **Linux 侧** (X2000 主核, XBurst@II.V2, 无 FPU) |
| 工具链 | mips-gcc720-glibc229 (GCC 7.2.0) |
| 编译标志 | `-O3 -march=mips32r2 -msoft-float -std=c99` |
| 降噪引擎 | **GTCRN 定点 (纯 C)** — 替换 NNoM denoise_stream_v8 |
| CRM SNR vs MATLAB | 58.8 dB (多帧) / 65.9 dB (单帧) |
| 推理耗时 | **28.03 ms/帧** @ X2000 Linux |
| 发布日期 | 2026-06-23 |

---

## iccom 数据流

```
RTOS (DSP 侧)                         Linux 侧
─────────────                         ────────
DMA 采集 8kHz PCM
  │
  ├─ 每 12.5ms: write(rf11, 100×int16)
  │                                      │
  │                                   read(rf11, buf)
  │                                      │
  │                                   环形缓冲累积
  │                                   → 满 200 样本触发推理
  │                                      │
  │                                   noise_reduction(in, out)
  │                                   耗时 ~28ms
  │                                      │
  │  ←─── write(wf12, 200×int16) ───     │
  │                                      │
  ├─ read(wf12, 200×int16)
  │
DMA 输出增强 PCM
```

**通道约定**: `rf11` (RTOS→Linux 音频下行) / `wf12` (Linux→RTOS 音频上行)

---

## API 接口

```c
#include "noise_reduction.h"

void noise_init(void);                              // 初始化 (启动时调用一次)
void noise_reduction(short *voiceIn, short *voiceOut); // 降噪 (每25ms调用)
void noise_deinit(void);                            // 销毁 (关机时可选)
```

### 参数规格

| 参数   | 值                                    |
| ---- | ------------------------------------ |
| 采样率  | **8kHz**                             |
| 帧长   | **200 样本** (25ms)                    |
| 数据类型 | `short` (int16)                      |
| 声道   | 单声道                                  |
| 延迟   | ~53ms (12.5ms缓冲 + 28ms推理 + 12.5ms输出) |

### 与旧库差异

| 参数 | 旧库 (DSP v8_agc) | **新库 (Linux GTCRN)** |
|------|:--:|:--:|
| 帧规格 | 100 样本/12.5ms | **200 样本/25ms** |
| 降噪引擎 | NNoM GRU (int8) | **GTCRN int32 Q20** |
| 内存 | ~12KB | **~420KB** |
| AGC | 内置 | **无** (纯降噪) |

---

## 内部流水线

```
noise_reduction(in, out)
  │
  ├─ 2× 上采样 (线性插值): 200×8kHz → 400×16kHz
  ├─ STFT (512-pt FFT, 256 hop, Hann窗, ±256 镜像padding)
  ├─ GTCRN 推理: gtcrn_infer_frame() → CRM (增强频谱)
  ├─ ISTFT (overlap-add, Hann窗)
  └─ 2× 下采样 (均值抽取): 16kHz → 200×8kHz
```

### GTCRN 内部管线

```
mag_gen → BM → SFE → Conv0 → Conv1
  → GT_Conv0(dil=1) → GT_Conv1(dil=2) → GT_Conv2(dil=5)
  → GDPRNN1 (Intra_BiGRU + Inter_GRU)
  → GDPRNN2 (Intra_BiGRU + Inter_GRU)
  → GT_DeConv0(dil=5) → GT_DeConv1(dil=2) → GT_DeConv2(dil=1)
  → DeConv1 → DeConv0 (Tanh→s16f15)
  → BS → MASK → CRM (2×257 s32f20)
```

---

## Linux 工程集成

### 1. 拷贝文件

```sh
cp noise_reduction.h          → ${PROJECT}/include/
cp gtcrn_fp.h                 → ${PROJECT}/include/
cp gtcrn_fp.c                 → ${PROJECT}/src/
cp gtcrn_infer.c              → ${PROJECT}/src/
cp gtcrn_matlab_weights.h     → ${PROJECT}/include/
```

### 2. Makefile

```makefile
CFLAGS  += -I$(PROJECT_DIR)/include -O3 -march=mips32r2 -msoft-float -std=c99
LDFLAGS += -lm -lpthread

GTCRN_SRC = src/gtcrn_fp.c src/gtcrn_infer.c src/noise_reduction.c
OBJS     += $(GTCRN_SRC:.c=.o)
```

### 3. 音频处理线程

```c
#include "noise_reduction.h"
#include <fcntl.h>
#include <unistd.h>

#define FRAME_IN  200   // 25ms @ 8kHz
#define CHUNK     100   // RTOS 每次发送 100 样本 (12.5ms)

static short ring_buf[FRAME_IN * 2];  // 双缓冲
static int   ring_pos = 0;

void *audio_thread(void *arg) {
    int fd_rf11 = open("/dev/iccom_rf11", O_RDONLY);
    int fd_wf12 = open("/dev/iccom_wf12", O_WRONLY);
    short in[FRAME_IN], out[FRAME_IN];

    noise_init();

    while (1) {
        short chunk[CHUNK];
        read(fd_rf11, chunk, CHUNK * sizeof(short));  // 阻塞等待 RTOS 数据

        // 环形缓冲累积
        memcpy(&ring_buf[ring_pos], chunk, CHUNK * sizeof(short));
        ring_pos += CHUNK;

        if (ring_pos >= FRAME_IN) {
            memcpy(in, ring_buf, FRAME_IN * sizeof(short));
            memmove(ring_buf, &ring_buf[FRAME_IN], (ring_pos - FRAME_IN) * sizeof(short));
            ring_pos -= FRAME_IN;

            noise_reduction(in, out);                   // 降噪 (~28ms)
            write(fd_wf12, out, FRAME_IN * sizeof(short));
        }
    }
}
```

### 4. 编译

```sh
cd ${PROJECT}
make clean && make
```

---

## 内存预算

| 组件 | 大小 |
|------|------|
| gtcrn_state_t | ~70 KB |
| 权重 (271 tensors) | ~302 KB |
| 栈 + 临时缓冲 | ~30 KB |
| STFT 环形缓冲 | ~10 KB |
| **总计** | **~412 KB** |

---

## 验证清单

| 项目 | 方法 | 预期 |
|------|------|------|
| 编译 | make clean && make | 0 error |
| 单帧 CRM | vs MATLAB golden | SNR > 55 dB |
| 多帧音频 | vs MATLAB enhanced | SNR > 55 dB |
| 实时性 | 计时器测 noise_reduction | < 25ms |
| 音频质量 | 对讲机实听 | 噪声抑制 + 人声清晰 |

---

**交付日期**: 2026-06-23
**源码路径**: `x2000_deploy_v2/linux_api/`
**关联 Session**: [[2026-06-22-003]], [[2026-06-23-002]]
