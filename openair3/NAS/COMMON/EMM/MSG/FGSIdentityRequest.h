#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "NR_NAS_defs.h"
#include "nas_log.h"
#include "TLVDecoder.h"

#ifndef FGS_IDENTITY_REQUEST_H_
#define FGS_IDENTITY_REQUEST_H_

int decode_fgs_identity_request(Identityrequest_t *fgs_identity_req, uint8_t *buffer, uint32_t len);

#endif /* FGS IDENTITY REQUEST_H_*/