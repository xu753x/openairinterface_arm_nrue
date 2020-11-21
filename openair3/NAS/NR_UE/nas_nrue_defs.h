#ifndef __NAS_NRUE_DEFS_H__
#define __NAS_NRUE_DEFS_H__

#include "NR_NAS_defs.h"

typedef struct  {
uint8_t iei;
uint8_t len1;
uint8_t len2;
uint8_t mi:8;
} noidentity_identity_response_msg;

#endif /* __NAS_NRUE_DEFS_H__*/