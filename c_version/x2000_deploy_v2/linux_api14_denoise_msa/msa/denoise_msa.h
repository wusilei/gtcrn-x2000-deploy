/**
 * denoise_msa.h — MSA-accelerated function declarations
 *
 * All functions accept only integer types — safe to link
 * hard-float MSA .o files with soft-float main binary.
 */
#ifndef DENOISE_MSA_H
#define DENOISE_MSA_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* BM_fixed: Band Merging (3,257)→(3,129).  1.87× speedup on X2000. */
void bm_fixed_msa(const int32_t *x, const uint16_t *weight, int32_t *y);

#ifdef __cplusplus
}
#endif
#endif
