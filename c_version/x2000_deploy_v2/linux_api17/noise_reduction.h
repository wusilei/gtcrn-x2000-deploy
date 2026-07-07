#ifndef NOISE_REDUCTION_H
#define NOISE_REDUCTION_H
#ifdef __cplusplus
extern "C" {
#endif
void noise_init(void);
void noise_reduction(short *voiceIn, short *voiceOut);
void noise_deinit(void);
#ifdef __cplusplus
}
#endif
#endif
