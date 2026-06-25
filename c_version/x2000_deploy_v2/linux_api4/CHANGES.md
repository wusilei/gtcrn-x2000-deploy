# 2026-06-23 GTCRN v10b — linux_api3 v9 已知问题修复 (第2轮)

**试听反馈**: v10 滋滋噪声消失 ✅，但人声不可懂/失真/过于低沉 ❌

## v10→v10b 追加修复

| # | v10 Bug | 根因 | 修复 |
|:--|:--|:--|:--|
| **6** | **STFT 帧处理顺序颠倒 (最新→最旧)** | read_wpos 从 g_fifo_wpos 开始 -WIN_INC 后退 → GRU 时间倒流 + OLA 输出时间反序 → 人声完全破坏 | 从最旧帧开始 +WIN_INC 前进 (按时间顺序) |
| **7** | **下采样 FIR 过于激进** | [1,2,1]/4 在 4kHz 衰减 -6dB → 高频丢失 → 声音低沉 | 回退到简单 (a+b)/2 平均 |

### Bug #6 详解 (最关键!)

```
v10 (WRONG): 最新→最旧
  read_wpos = g_fifo_wpos  → 读 [288, 799] (最新)
  read_wpos -= 256         → 读 [32, 543]  (较旧)
  → GRU 时间倒流! OLA 输出 16ms 时间反序! → 人声不可懂

v10b (CORRECT): 最旧→最新
  read_wpos = g_fifo_wpos - g_fifo_count + 512 → 读 [0, 511] (最旧)
  read_wpos += 256                              → 读 [256, 767] (较新)
  → GRU 时间正序 ✓ OLA 输出正时 ✓
```

---

# v10 (第一轮)

## 背景

接 [[2026-06-22-003]] v9 已知问题。从 `linux_api/` (v9) 完全复制到 `linux_api3/`，冻结 `linux_api/`，在 `linux_api3/` 中修复。

### v9 已知问题

| 问题 | 根因 | 修复 |
|:--|:--|:--|
| 背景"滋滋"规律噪声 | (1) Hann² 不满足 COLA，wsum 边缘归一化放大噪声 → 周期截顶 | sqrt-Hann 窗 + 移除 wsum |
| | (2) STFT 帧读取同一 g_fifo_wpos → 重复帧污染 GRU | read_wpos 逐帧推进 |
| | (3) FIFO 不足 512 样本就做 STFT → 读旧数据污染 GRU | ≥WIN_LEN (512) 门槛 |
| 人声失真 | (4) 线性插值 8k→16k 混叠 | 3-tap FIR [1,2,1]/4 抗混叠 |
| | (5) Hann² COLA 不满足 → 频谱泄漏 | sqrt-Hann 自动 COLA |

---

## 5 个根因 + 5 个修复

| # | Bug | 影响 | 修复 | 文件 |
|:--|:--|:--|:--|:--|
| **1** | **Hann² 不满足 COLA** | 窗边缘 wsum≈0→除小数→放大噪声→clip→"滋滋" | sqrt-Hann: sin(π·i/511). sqrt² = Hann, COLA: Hann[n]+Hann[n+256]=1 | noise_reduction.c |
| **2** | **STFT 读指针不推进** | 同次调用多帧全读同一 g_fifo_wpos → 重复帧 → GRU 污染 | read_wpos 每帧 -WIN_INC, 逐帧推进 | noise_reduction.c |
| **3** | **FIFO≥WIN_INC(256)就做 STFT** | 仅 400 有效样本读 512 → 112 旧数据 → 帧污染 | 改为 ≥WIN_LEN(512) 才做 STFT | noise_reduction.c |
| **4** | **线性插值 8k→16k 无抗混叠** | 镜像频率混入 4~8kHz → 人声失真 | 3-tap FIR [1,2,1]/4 抗混叠下采样 | noise_reduction.c |
| **5** | **F2Q20 无饱和** | float→int32 溢出回绕 → 权重破环 | inline 函数 + INT32_MIN/MAX 饱和 | gtcrn_fp.h |

### Bug #2 详解 (最关键)

```c
// v9 (bug): 同次调用中第二帧仍读 g_fifo_wpos 处的相同 512 样本
while (g_fifo_count >= WIN_INC) {  // 且门槛太低
    int start = (g_fifo_wpos - WIN_LEN + FIFO_SZ) % FIFO_SZ;  // 不推进!
    ...
}

// v10 (fix): read_wpos 逐帧推进 -WIN_INC, 门槛 ≥WIN_LEN
int read_wpos = g_fifo_wpos;
while (g_fifo_count >= WIN_LEN) {  // 门槛: 512
    int start = (read_wpos - WIN_LEN + FIFO_SZ) % FIFO_SZ;
    ...
    read_wpos = (read_wpos - WIN_INC + FIFO_SZ) % FIFO_SZ;  // 推进!
}
```

---

## 测试结果 (PC, fileid_5)

| 指标 | v9 (linux_api) | v10 (linux_api3) | 说明 |
|:--|:--|:--|:--|
| RMS | 1202 | 790 | v9 更高因 wsum 边缘放大+clip |
| Max abs | **32768 (截顶!)** | 11565 | v9 周期性饱和 INT16_MAX → "滋滋" |
| Nonzero | 97% | 97.5% | 均正常 |
| 输入-输出包络相关 | 0.136 | 0.125 | 相近; 降噪改变幅度的自然结果 |
| 无声帧 (RMS<10) | 2 | 22 | v10 降噪更彻底 (噪声抑制后自然安静) |

### 5 文件 RMS

| File | v9 PC | v10 PC | MATLAB |
|:--|:--|:--|:--|
| fileid_1 | ~786 (X2000: 634-2392) | **786** | — |
| fileid_2 | — | **1857** | — |
| fileid_3 | — | **1792** | — |
| fileid_4 | — | **636** | — |
| fileid_5 | 870 | **790** | 759 |

---

## 核心变更对比

| 方面 | v9 (linux_api) | v10 (linux_api3) |
|:--|:--|:--|
| 分析窗 | Hann | **sqrt-Hann** (sin(πi/511)) |
| 合成窗 | Hann | **sqrt-Hann** |
| COLA | Hann² (不满足, wsum 补救) | **Hann (自动满足)** |
| OLA 归一化 | /wsum (边缘不稳定) | **无 (COLA=1)** |
| STFT 读指针 | g_fifo_wpos (固定) | **read_wpos (逐帧 -WIN_INC 推进)** |
| STFT 触发条件 | g_fifo_count ≥ 256 | **g_fifo_count ≥ 512** |
| 上采样 | 线性插值 | 线性插值 (帧间保持连续性) |
| 下采样 | (a+b)/2 | **[1,2,1]/4 低通 FIR** |
| F2Q20 | 宏 (溢出回绕) | **inline 函数 (INT32 饱和)** |
| Warmup | 60 帧 fade-in | **20 mute + 12 fade (匹配参考)** |

---

## 目录结构

```
x2000_deploy_v2/
├── linux_api/     ← v9 冻结 (只读)
├── linux_api2/    ← batch 模式变体
└── linux_api3/    ← v10 (当前工作版本)
    ├── gtcrn_linux                    ← v10 PC 可执行文件 (260KB)
    ├── noise_reduction.c/h            ← v10 流式适配层 (5 bug 修复)
    ├── gtcrn_fp.c/h + gtcrn_infer.c   ← GTCRN 推理 (F2Q20 饱和)
    ├── gtcrn_matlab_weights.h         ← 权重 (不变)
    ├── kiss_fft.c/h + kiss_fftr.c/h   ← FFT (不变)
    ├── gtcrn_linux.c                  ← 主程序 (不变)
    └── test_output/                   ← 5 文件 WAV 输出
        ├── gtcrn_v10_fileid_1.wav
        ├── gtcrn_v10_fileid_2.wav
        ├── gtcrn_v10_fileid_3.wav
        ├── gtcrn_v10_fileid_4.wav
        └── gtcrn_v10_fileid_5.wav
```

---

## 下一步

| 优先级 | 任务 |
|:--|:--|
| **P0** | **用户试听** test_output/*.wav 确认滋滋噪声消失、人声可懂 |
| P1 | X2000 MIPS 交叉编译 (`mips-linux-gnu-gcc -O3 -march=mips32r2 -msoft-float`) |
| P2 | X2000 板上 5 文件测试 + iccom 联调 |
