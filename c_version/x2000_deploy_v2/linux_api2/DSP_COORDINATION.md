# GTCRN Linux 侧 — DSP 联调指南

## 编译好的二进制

```
x2000_deploy_v2/linux_api/gtcrn_linux  (878KB, MIPS32R2, 静态链接)
```

## iccom 通道约定

| 方向 | 设备 | 格式 | 频率 |
|:--|:--|:--|:--|
| RTOS → Linux | `/dev/iccom_rf11` | 200×int16 @ 8kHz | 每 25ms |
| Linux → RTOS | `/dev/iccom_wf12` | 200×int16 @ 8kHz | 每 25ms |

> 帧长 200 样本 = 25ms @ 8kHz，与 GTCRN 推理耗时 28ms 匹配。

## DSP 工程师需要做的

### 1. RTOS 侧: 发送原始音频到 Linux

```c
// 每 25ms 在 RTOS 音频 ISR 中调用:
short frame[200];
dma_read(frame, 200);                              // 从 codec DMA 读取 200 样本
iccom_write(ICCOM_RF11, frame, 200 * sizeof(short)); // 发送给 Linux
```

### 2. RTOS 侧: 接收增强音频从 Linux

```c
// 轮询或中断方式读取 wf12:
short enhanced[200];
int nr = iccom_read(ICCOM_WF12, enhanced, 200 * sizeof(short));
if (nr == 200 * sizeof(short)) {
    dma_write(enhanced, 200);                      // 输出到 codec
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
scp gtcrn_linux root@192.168.42.159:/usr/bin/
chmod +x /usr/bin/gtcrn_linux
```

### 2. 启动降噪服务

```sh
# 前台运行 (调试):
/usr/bin/gtcrn_linux

# 后台运行 (生产):
/usr/bin/gtcrn_linux &
```

### 3. 自测模式 (不依赖 RTOS, stdin→GTCRN→stdout)

```sh
# PC 上生成测试 PCM:
python3 -c "
import numpy as np
audio = np.fromfile('noisy_fileid_5.wav', dtype=np.int16, offset=44)
audio[:20000].tofile('test_in.pcm')
"

# X2000 上:
./gtcrn_linux test < test_in.pcm > test_out.pcm

# 检查输出 (前 2000 字节是暖机静音, 之后有数据):
xxd test_out.pcm | tail -20
```

### 4. 验证流水线

```sh
# 方法 A: 示波器看 wf12 有无数据输出
# 方法 B: cat /dev/iccom_wf12 | xxd   (看是否有非零 PCM)
# 方法 C: 对讲机实际试听
```

## 调试步骤 (按顺序)

### Step 1: 确认 iccom 通道通

```sh
# DSP 侧持续发送测试数据 (如 ramp 0,1,2,...):
# Linux 侧:
cat /dev/iccom_rf11 | xxd | head
# 应看到递增的数据流
```

### Step 2: 启动 GTCRN, 验证输入

```sh
# 修改 gtcrn_linux.c: 打印 rf11 读取的前 10 个样本
# 重新编译部署, 确认数据到达
```

### Step 3: 验证输出

```sh
# DSP 侧接收 wf12 数据, 打印前 10 个样本
# 前 ~2000 字节是暖机静音(全零), 之后应有非零增强音频
```

### Step 4: 实听

```
对讲机开机 → DSP 启动 rf11 发送 → Linux 启动 gtcrn_linux → wf12 接收 → DAC 输出
```

## 暖机说明

首 ~40 帧 (约 1 秒) 输出全零/衰减，GRU 状态收敛中。之后降噪正常。

## 故障排查

| 现象 | 可能原因 | 检查 |
|------|------|------|
| rf11 open 失败 | 设备未创建 | `ls /dev/iccom_rf11` |
| read 阻塞不返回 | RTOS 未发送数据 | DSP 侧确认 rf11 write |
| 输出全零持续 | STFT 缓冲未满或 GRU 未收敛 | 等待 2 秒 |
| 声音失真 | STFT 参数不匹配 | 确认 8kHz 采样率 |
| CPU 100% | 推理耗时超标 | `time gtcrn_linux test` 测耗时 |
