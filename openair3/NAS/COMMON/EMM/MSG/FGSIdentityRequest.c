/*! \file FGSIdentityRequest.c

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "nas_log.h"

#include "FGSIdentityRequest.h"


int decode_fgs_identity_request(Identityrequest_t *fgs_identity_req, uint8_t *buffer, uint32_t len)
{
  int decoded = 0;

  fgs_identity_req->it = *(buffer + decoded);
  decoded++;

  return decoded;
}


