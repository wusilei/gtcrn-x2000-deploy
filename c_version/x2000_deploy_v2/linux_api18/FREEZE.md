# linux_api14 — FROZEN

**Frozen on**: 2026-06-26
**Reason**: linux_api14_denoise (new denoise model) produces buzzing noise + no voice on X2000. Root cause: denoise model outputting invalid audio. linux_api14 GTCRN model works correctly.

**Binary**: `gtcrn_linux_q15` (841884 bytes, MIPS32R2 static)
**Deployed as**: `/data/gtcrn_v20_q15` on X2000

**Pipeline**:
- iccom rf11→wf12, 400×int16 @ 50ms, 8kHz
- Pre-AGC → GTCRN → output
- Float KissFFT forward, Q15 fft_q15 inverse
- sqrt-Hann window, OLA

**Key files**:
- gtcrn_linux.c — iccom main loop + energy detection + AGC
- noise_reduction_q15.c — STFT/ISTFT pipeline (called by gtcrn_linux.c)
- gtcrn_fp.c/h — GTCRN model inference (fixed-point)
- gtcrn_infer.c — gtcrn_infer_frame() wrapper
- gtcrn_matlab_weights.h — model weights
- fft_q15.h — Q15 512-pt inverse FFT
- kiss_fft.c/h — float forward FFT
- agc.c/h — voice_AGC LMS

**Build command**:
```
mips-linux-gnu-gcc -O3 -std=c99 -march=mips32r2 -msoft-float \
  -o gtcrn_linux_q15 agc.c gtcrn_linux.c noise_reduction_q15.c \
  gtcrn_fp.c gtcrn_infer.c -lm -lrt -static
```

**DO NOT MODIFY** — this is the production baseline.
