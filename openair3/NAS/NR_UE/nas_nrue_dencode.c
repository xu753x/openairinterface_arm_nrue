
// #include "nas_nrue_dencode.h"

// int encode_IdentityresponseIMSI(IdentityresponseIMSI_t *identity_response, uint8_t *buffer, uint32_t len)
// {
//   int encoded = 0;
//   int encode_result = 0;

//   /* Checking IEI and pointer */
//   // CHECK_PDU_POINTER_AND_LENGTH_ENCODER(buffer, IDENTITY_RESPONSE_MINIMUM_LENGTH, len);

//   if ((encode_result =
//          encode_mobileidentity(identity_response, 0, buffer +
//                                 encoded, len - encoded)) < 0)        //Return in case of error
//     return encode_result;
//   else
//     encoded += encode_result;

//   return encoded;
// }

// int encode_authenticationresponse(authenticationresponse_t *authentication_response, uint8_t *buffer, uint32_t len)
// {
//   int encoded = 0;
//   int encode_result = 0;

//   /* Checking IEI and pointer */
//   CHECK_PDU_POINTER_AND_LENGTH_ENCODER(buffer, AUTHENTICATION_RESPONSE_MINIMUM_LENGTH, len);

//   if ((encode_result =
//          encode_authentication_response_parameter(&authentication_response->authenticationresponseparameter,
//              0, buffer + encoded, len - encoded)) < 0)        //Return in case of error
//     return encode_result;
//   else
//     encoded += encode_result;

//   return encoded;
// }

// int encode_mobileidentity(IdentityresponseIMSI_t *identity_response, uint8_t iei, uint8_t *buffer, uint32_t len)
// {
//   uint8_t *lenPtr;
//   int encoded_rc = TLV_ENCODE_VALUE_DOESNT_MATCH;
//   uint32_t encoded = 0;
// //   /* Checking IEI and pointer */
// //   CHECK_PDU_POINTER_AND_LENGTH_ENCODER(buffer, MOBILE_IDENTITY_MINIMUM_LENGTH, len);
// // #if defined (NAS_DEBUG)
// //   dump_mobile_identity_xml(mobileidentity, iei);
// // #endif

//   if (iei > 0) {
//     *buffer = iei;
//     encoded++;
//   }

//   lenPtr  = (buffer + encoded);
//   encoded ++;


//   if (identity_response->mi == SUCI) {
//     encoded_rc = encode_suci_identity(identity_response,
//                   buffer + encoded);
//   }

//   if (encoded_rc > 0) {
//     *lenPtr = encoded + encoded_rc - 1 - ((iei > 0) ? 1 : 0);
//   }

//   if (encoded_rc < 0) {
//     return encoded_rc;
//   }

//   return (encoded + encoded_rc);
// }

// static int encode_suci_identity(IdentityresponseIMSI_t *imsi, uint8_t *buffer)
// {
//   uint32_t encoded = 0;
//   *(buffer + encoded) = 0x00 | (imsi->supiFormat << 4) | imsi->identityType;
//   encoded++;
//   *(buffer + encoded) = 0x00 | (imsi->mcc2 << 4) | imsi->mcc1;
//   encoded++;
//   *(buffer + encoded) = 0x00 | (imsi->mnc3 << 4) | imsi->mcc3;
//   encoded++;
//   *(buffer + encoded) = 0x00 | (imsi->mnc2 << 4) | imsi->mnc1;
//   encoded++;
//   *(buffer + encoded) = 0x00 | (imsi->routing2 << 4) | imsi->routing1;
//   encoded++;
//   *(buffer + encoded) = 0x00 | (imsi->routing4 << 4) | imsi->routing3;
//   encoded++;
//   *(buffer + encoded) = 0x00 | (imsi->spare << 4) | imsi->protectScheme;
//   encoded++;
//   //QUES：encode_suci_identity
//   // Home network public key identifier
//   // Scheme output 没写
//   // return encoded;
//   // 定义的hplmnId nocore没用 
// }

int encode_noidentity_response (noidentity_identity_response_msg *identity_response, uint8_t *buffer, uint32_t len)
{
  int encoded = 0;
  int encode_result = 0;
  if (identity_response->iei> 0) {
    *buffer = identity_response->iei;
    encoded++;
  }
  *(buffer + encoded)=0;
  encoded++;
  *(buffer + encoded)=3;
  encoded++;
  *(buffer + encoded)=0;
  encoded++;
  return encoded;

}