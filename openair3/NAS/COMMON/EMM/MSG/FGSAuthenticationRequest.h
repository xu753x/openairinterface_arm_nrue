#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "NR_NAS_defs.h"
#include "nas_log.h"
#include "TLVDecoder.h"

#ifndef FGS_AUTHENTICATION_REQUEST_H_
#define FGS_AUTHENTICATION_REQUEST_H_

#define AUTHENTICATION_PARAMETER_RAND_IEI 0x21
#define AUTHENTICATION_PARAMETER_AUTN_IEI 0x20

int decode_fgs_authentication_request(authenticationrequestHeader_t *fgs_authentication_req, uint8_t *buffer, uint32_t len);

#endif /* FGS AUTHENTICATION REQUEST_H_*/