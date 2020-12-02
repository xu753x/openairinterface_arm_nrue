/*! \file FGSAuthenticationRequest.c

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "FGSAuthenticationRequest.h"


int decode_fgs_authentication_request(authenticationrequestHeader_t *fgs_authentication_req, uint8_t *buffer, uint32_t len)
{
  int decoded = 0;

  fgs_authentication_req->ngKSI = (*(buffer + decoded)>>4) & 0x0f;
  fgs_authentication_req->spare = *(buffer + decoded) & 0x0f;
  decoded++;

  IES_DECODE_U8(buffer, decoded, fgs_authentication_req->ABBALen);
  IES_DECODE_U16(buffer, decoded, fgs_authentication_req->ABBA);

  while(len - decoded > 0) {
      uint8_t ieiDecoded = *(buffer + decoded);
      switch(ieiDecoded) {
        case AUTHENTICATION_PARAMETER_RAND_IEI:
          IES_DECODE_U8(buffer, decoded, fgs_authentication_req->ieiRAND);
          memcpy(fgs_authentication_req->RAND, buffer+decoded, sizeof(fgs_authentication_req->RAND));
          decoded += sizeof(fgs_authentication_req->RAND);
          break;

        case AUTHENTICATION_PARAMETER_AUTN_IEI:
          IES_DECODE_U8(buffer, decoded, fgs_authentication_req->ieiAUTN);
          IES_DECODE_U8(buffer, decoded, fgs_authentication_req->AUTNlen);
          memcpy(fgs_authentication_req->AUTN, buffer+decoded, sizeof(fgs_authentication_req->AUTN));
          decoded += sizeof(fgs_authentication_req->AUTN);
        
        default:
          return TLV_DECODE_UNEXPECTED_IEI;
      }
  }

  return decoded;
}

