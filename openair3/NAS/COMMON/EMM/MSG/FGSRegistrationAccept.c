 /*! \file FGSRegistrationAccept.c

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "FGSRegistrationAccept.h"

int decode_fgs_tracking_area_identity_list(FGSTrackingAreaIdentityList *tailist, uint8_t iei, uint8_t *buffer, uint32_t len) {
  int decoded_rc = TLV_DECODE_VALUE_DOESNT_MATCH;
  int decoded = 0;
  uint8_t ielen = 0;

  if (iei > 0) {
    CHECK_IEI_DECODER(iei, *buffer);
    decoded++;
  }

  ielen = *(buffer + decoded);
  decoded ++;
  CHECK_LENGTH_DECODER(len - decoded, ielen);
  
  uint8_t typeoflist = *(buffer + decoded) & 0x60;

  if (typeoflist == TRACKING_AREA_IDENTITY_LIST_ONE_PLMN_NON_CONSECUTIVE_TACS) {

    tailist->typeoflist = (*(buffer + decoded) >> 5) & 0x3;
    tailist->numberofelements = *(buffer + decoded) & 0x1f;
    decoded++;
    tailist->mccdigit2 = (*(buffer + decoded) >> 4) & 0xf;
    tailist->mccdigit1 = *(buffer + decoded) & 0xf;
    decoded++;
    tailist->mncdigit3 = (*(buffer + decoded) >> 4) & 0xf;
    tailist->mccdigit3 = *(buffer + decoded) & 0xf;
    decoded++;
    tailist->mncdigit2 = (*(buffer + decoded) >> 4) & 0xf;
    tailist->mncdigit1 = *(buffer + decoded) & 0xf;
    decoded++;
    IES_DECODE_U24(buffer, decoded, tailist->tac);
    return decoded;
  } else {
    return decoded_rc;
  }
}

int decode_fgs_allowed_nssa(NSSAI *nssai, uint8_t iei, uint8_t *buffer, uint32_t len) {
  int decoded = 0;
  uint8_t ielen = 0;

  if (iei > 0) {
    CHECK_IEI_DECODER(iei, *buffer);
    decoded++;
  }

  ielen = *(buffer + decoded);
  decoded ++;
  CHECK_LENGTH_DECODER(len - decoded, ielen);
  
  IES_DECODE_U8(buffer, decoded, nssai->length);
  IES_DECODE_U8(buffer, decoded, nssai->value);

  return decoded;
}

int decode_fgs_network_feature_support(FGSNetworkFeatureSupport *fgsnetworkfeaturesupport, uint8_t iei, uint8_t *buffer, uint32_t len) {
  int decoded = 0;
  uint8_t ielen = 0;

  if (iei > 0) {
    CHECK_IEI_DECODER(iei, *buffer);
    decoded++;
  }

  ielen = *(buffer + decoded);
  decoded ++;
  CHECK_LENGTH_DECODER(len - decoded, ielen);

  fgsnetworkfeaturesupport->mpsi = (*(buffer + decoded)>>7) & 0x1;
  fgsnetworkfeaturesupport->iwkn26 = (*(buffer + decoded)>>6) & 0x1;
  fgsnetworkfeaturesupport->EMF = (*(buffer + decoded)>>4) & 0x3;
  fgsnetworkfeaturesupport->EMC = (*(buffer + decoded)>>2) & 0x3;
  fgsnetworkfeaturesupport->IMSVoPSN3GPP = (*(buffer + decoded)>>1) & 0x1;
  fgsnetworkfeaturesupport->IMSVoPS3GPP = *(buffer + decoded) & 0x1;
  decoded ++;

  fgsnetworkfeaturesupport->MCSI =  (*(buffer + decoded)>>1) & 0x1;
  fgsnetworkfeaturesupport->EMCN3 = *(buffer + decoded) & 0x1;
  decoded ++;

  return decoded;
}

int decode_fgs_registration_accept(fgs_registration_accept_msg *fgs_registration_acc, uint8_t *buffer, uint32_t len)
{
  int decoded = 0;
  int decode_result = 0;

  IES_DECODE_U8(buffer, decoded, fgs_registration_acc->fgsregistrationresult.resultlength);
  fgs_registration_acc->fgsregistrationresult.spare = (*(buffer + decoded)>>4) & 0xf;
  fgs_registration_acc->fgsregistrationresult.smsallowed = (*(buffer + decoded)>>3) & 0x1;
  fgs_registration_acc->fgsregistrationresult.registrationresult = *(buffer + decoded) & 0x7;
  decoded++;

  while(len - decoded > 0) {
      uint8_t ieiDecoded = *(buffer + decoded);
      switch(ieiDecoded) {
        case REGISTRATION_ACCEPT_MOBILE_IDENTITY:
          if ((decode_result = decode_5gs_mobile_identity(&fgs_registration_acc->fgsmobileidentity, REGISTRATION_ACCEPT_MOBILE_IDENTITY, buffer +
                                  decoded, len - decoded)) < 0) {       //Return in case of error
            return decode_result;
          } else {
            decoded += decode_result;
            break;
          }

        case REGISTRATION_ACCEPT_5GS_TRACKING_AREA_IDENTITY_LIST:
          if ((decode_result = decode_fgs_tracking_area_identity_list(&fgs_registration_acc->tailist, REGISTRATION_ACCEPT_5GS_TRACKING_AREA_IDENTITY_LIST, buffer +
                                  decoded, len - decoded)) < 0) {       //Return in case of error
            return decode_result;
          } else {
            decoded += decode_result;
            break;
          }

        case REGISTRATION_ACCEPT_ALLOWED_NSSA:
          if ((decode_result = decode_fgs_allowed_nssa(&fgs_registration_acc->nssai, REGISTRATION_ACCEPT_ALLOWED_NSSA, buffer +
                                  decoded, len - decoded)) < 0) {       //Return in case of error
            return decode_result;
          } else {
            decoded += decode_result;
            break;
          }

        case REGISTRATION_ACCEPT_5GS_NETWORK_FEATURE_SUPPORT:
          if ((decode_result = decode_fgs_network_feature_support(&fgs_registration_acc->fgsnetworkfeaturesupport, REGISTRATION_ACCEPT_5GS_NETWORK_FEATURE_SUPPORT, buffer +
                                  decoded, len - decoded)) < 0) {       //Return in case of error
            return decode_result;
          } else {
            decoded += decode_result;
            break;
          }
        
        default:
          return TLV_DECODE_UNEXPECTED_IEI;
      }
  }

  return decoded;
}