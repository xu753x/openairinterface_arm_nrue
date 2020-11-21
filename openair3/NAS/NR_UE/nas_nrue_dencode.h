#ifndef __NAS_NRUE_DECODE_H__
#define __NAS_NRUE_DECODE_H__

#include "nas_nrue_defs.h"

int encode_noidentity_response (noidentity_identity_response_msg *identity_response, uint8_t *buffer, uint32_t len);

#endif /* __NAS_NRUE_DECODE_H__*/
