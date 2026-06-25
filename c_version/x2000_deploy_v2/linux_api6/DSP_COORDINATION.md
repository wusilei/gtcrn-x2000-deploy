# GTCRN Linux 侧 — DSP 联调指南 (v11 final, 50ms iccom)

## 编译好的二进制

```
x2000_deploy_v2/linux_api4/gtcrn_linux_mips  (880KB, MIPS32R2, 静态链接)
```

## iccom 通道约定

| 方向 | 设备 | 格式 | 频率 |
|:--|:--|:--|:--|
| RTOS → Linux | `/dev/iccom_rf11` | 400×int16 @ 8kHz | 每 **50ms** |
| Linux → RTOS | `/dev/iccom_wf12` | 400×int16 @ 8kHz | 每 **50ms** |

> Linux 侧内部将 400 样本拆为 2×200，`noise_reduction()` API 保持 25ms/次。接口不变，仅 iccom 读/写粒度增大。

## DSP 工程师需要做的

### 1. RTOS 侧: 发送原始音频到 Linux

```c
// 每 50ms 在 RTOS 音频 ISR 中调用:
short frame[400];                                     // 50ms @ 8kHz
dma_read(frame, 400);                                 // 从 codec DMA 读取 400 样本
iccom_write(ICCOM_RF11, frame, 400 * sizeof(short));  // 发送给 Linux
```

### 2. RTOS 侧: 接收增强音频从 Linux

```c
// 轮询或中断方式读取 wf12 (50ms 周期):
short enhanced[400];
int nr = iccom_read(ICCOM_WF12, enhanced, 400 * sizeof(short));
if (nr == 400 * sizeof(short)) {
    dma_write(enhanced, 400);                         // 输出到 codec
}
```

### 3. 确认 iccom 设备已创建

```sh
ls -la /dev/iccom_rf11 /dev/iccom_wf12
# 应存在, 权限 crw-rw---- root:root
```

## Linux 工程师需要做的

### 1. 部署二进制

```sh
scp linux_api4/gtcrn_linux_mips root@192.168.42.159:/usr/bin/gtcrn_linux
chmod +x /usr/bin/gtcrn_linux
```

### 2. 启动降噪服务

```sh
# 前台运行 (调试):
/usr/bin/gtcrn_linux

# 后台运行 (生产):
/usr/bin/gtcrn_linux &
```

启动后输出: `GTCRN Linux — iccom mode (rf11→GTCRN→wf12, 400smp/50ms)` 然后 `GTCRN ready. Waiting for audio from rtos...`

### 3. 自测模式

```sh
# test 模式: stdin 8kHz PCM → GTCRN → stdout 8kHz PCM (对讲机实录)
./gtcrn_linux test < test_8k.pcm > test_out_8k.pcm

# test16k 模式: stdin 16kHz PCM → 降采样8kHz → GTCRN → stdout 8kHz PCM (MATLAB导出)
./gtcrn_linux test16k < test_16k.pcm > test_out_8k.pcm
```

```sh
# PC 上生成测试 PCM:
python3 -c "
import numpy as np
audio = np.fromfile('noisy_fileid_5.wav', dtype=np.int16, offset=44)
audio[:20000].tofile('test_in.pcm')
"

# X2000 上:
./gtcrn_linux test16k < test_in.pcm > test_out.pcm

# 检查输出 (前 ~2000 字节是暖机静音, 之后有数据):
xxd test_out.pcm | tail -20
```

### 4. 验证流水线

```sh
# 方法 A: 示波器看 wf12 有无数据输出 (每 50ms 400 sample)
# 方法 B: cat /dev/iccom_wf12 | xxd   (看是否有非零 PCM)
# 方法 C: 对讲机实际试听
```

## Linux 侧内部流程

```
rf11 read(400×int16) ─→ noise_reduction(in[0..199],   out[0..199])   ← 25ms 帧 #1
                      ─→ noise_reduction(in[200..399], out[200..399]) ← 25ms 帧 #2
                      ─→ wf12 write(400×int16)
```

`noise_reduction()` 内部:
1. 8kHz→16kHz 上采样 (线性插值)
2. FIFO 缓冲 → STFT (512-pt, 256 hop, sqrt-Hann 窗)
3. GTCRN 定点推理 (~28ms/帧)
4. IFFT → OLA (COLA 自动满足) → 高频 EQ → 下采样 → 8kHz PCM

## 调试步骤 (按顺序)

### Step 1: 确认 iccom 通道通

```sh
# DSP 侧持续发送测试数据 (如 ramp 0,1,2,...):
# Linux 侧:
cat /dev/iccom_rf11 | xxd | head
# 应看到递增的数据流 (每 50ms 400×int16 = 800 bytes)
```

### Step 2: 启动 GTCRN, 验证输入

```sh
# 修改 gtcrn_linux.c: 打印 rf11 读取的前 10 个样本和读取字节数
# 每 50ms 应打印 800 bytes, 重新编译部署
```

### Step 3: 验证输出

```sh
# DSP 侧接收 wf12 数据, 打印前 10 个样本
# 前 ~2000 字节是暖机静音(全零), 之后应有非零增强音频
# 每 50ms 应收到 800 bytes (400×int16)
```

### Step 4: 实听

```
对讲机开机 → DSP 启动 rf11 发送(400/50ms) → Linux 启动 gtcrn_linux → wf12 接收(400/50ms) → DAC 输出
```

## 暖机说明

首 ~20 个 iccom 周期 (约 1 秒, 40 帧×25ms) 输出全零/淡入，GRU 状态收敛中。之后降噪正常。

## 故障排查

| 现象 | 可能原因 | 检查 |
|------|------|------|
| rf11 open 失败 | 设备未创建 | `ls /dev/iccom_rf11` |
| rf11 short read (<800 bytes) | RTOS 发送数据不足 400 sample | DSP 侧确认 rf11 write 长度 |
| read 阻塞不返回 | RTOS 未发送数据 | DSP 侧确认 rf11 write |
| wf12 short write | DSP 侧接收缓冲不足 | 确认 DSP 侧 read 400×int16 |
| 输出全零持续 | STFT 缓冲未满或 GRU 未收敛 | 等待 2 秒 |
| 声音失真 | 采样率不匹配 | 确认 8kHz, 400 sample/50ms |
| rf11 read 阻塞超 50ms | Linux 侧 2 帧推理耗时长 | 见性能说明 |
| CPU 100% | 推理耗时超标 | `time gtcrn_linux test` 测耗时 |

## 性能说明

- 单帧 `noise_reduction` 耗时 ~47.5ms @ X2000 (含上/下采样 + STFT + GTCRN + ISTFT)
- iccom 50ms 周期内需完成 **2 帧** (2×47.5=95ms)，当前超出 50ms 窗口
- **P1 优化目标**: 每帧 <25ms，使 2 帧总耗时 <50ms
- 优化方向: FFT 定点化、上/下采样简化、编译器调优

## 版本对照

| 目录 | 版本 | iccom 帧 | 状态 |
|:--|:--|:--|:--|
| `linux_api/` | v9 | 200/25ms | ❄️ 冻结 |
| `linux_api2/` | batch 变体 | — | ❄️ 冻结 |
| `linux_api3/` | v10 | 200/25ms | ❄️ 冻结 |
| `linux_api4/` | **v11 final** | **400/50ms** | ✅ 当前交付 |
