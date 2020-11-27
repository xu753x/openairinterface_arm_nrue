/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file FGSMobileIdentity.c
 * \brief 5GS Mobile Identity for registration request procedures
 * \author Yoshio INOUE, Masayuki HARADA
 * \email yoshio.inoue@fujitsu.com,masayuki.harada@fujitsu.com
 * \date 2020
 * \version 0.1
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


#include "TLVEncoder.h"
#include "TLVDecoder.h"
#include "FGSMobileIdentity.h"

static int decode_guti_5gs_mobile_identity(Guti5GSMobileIdentity_t *guti, uint8_t *buffer);

static int encode_guti_5gs_mobile_identity(Guti5GSMobileIdentity_t *guti, uint8_t *buffer);
static int encode_suci_5gs_mobile_identity(Suci5GSMobileIdentity_t *suci, uint8_t *buffer);
static int encode_imeisv_5gs_mobile_identity(Imeisv5GSMobileIdentity_t *imeisv, uint8_t *buffer);

int decode_5gs_mobile_identity(FGSMobileIdentity *fgsmobileidentity, uint8_t iei, uint8_t *buffer, uint32_t len)
{
  int decoded_rc = TLV_DECODE_VALUE_DOESNT_MATCH;
  int decoded = 0;
  uint8_t ielen = 0;

  if (iei > 0) {
    CHECK_IEI_DECODER(iei, *buffer);
    decoded++;
  }

  ielen = *(buffer + decoded);
  decoded++;
  CHECK_LENGTH_DECODER(len - decoded, ielen);

  uint8_t typeofidentity = *(buffer + decoded) & 0x7;

  if (typeofidentity == FGS_MOBILE_IDENTITY_5G_GUTI) {
    decoded_rc = decode_guti_5gs_mobile_identity(&fgsmobileidentity->guti,
                 buffer + decoded);
  }

  if (decoded_rc < 0) {
    return decoded_rc;
  }

  return (decoded + decoded_rc);
}

int encode_5gs_mobile_identity(FGSMobileIdentity *fgsmobileidentity, uint8_t iei, uint8_t *buffer, uint32_t len)
{
  int encoded_rc = TLV_ENCODE_VALUE_DOESNT_MATCH;
  uint32_t encoded = 0;

  if (iei > 0) {
    *buffer = iei;
    encoded++;
  }

  encoded = encoded + 2;

  if (fgsmobileidentity->guti.typeofidentity == FGS_MOBILE_IDENTITY_5G_GUTI) {
    encoded_rc = encode_guti_5gs_mobile_identity(&fgsmobileidentity->guti,
                 buffer + encoded);
  }

  if (fgsmobileidentity->suci.typeofidentity == FGS_MOBILE_IDENTITY_SUCI) {
    encoded_rc = encode_suci_5gs_mobile_identity(&fgsmobileidentity->suci,
                 buffer + encoded);
  }

  if (fgsmobileidentity->imeisv.typeofidentity == FGS_MOBILE_IDENTITY_IMEISV) {
    encoded_rc = encode_imeisv_5gs_mobile_identity(&fgsmobileidentity->imeisv,
                 buffer + encoded);
  }

  if (encoded_rc < 0) {
    return encoded_rc;
  }

  if(iei > 0){
    *(uint16_t*) (buffer+1) = htons(encoded  + encoded_rc - 3);
  } else {
    *(uint16_t*) buffer = htons(encoded  + encoded_rc - 2);
  }

  return (encoded + encoded_rc);
}

static int decode_guti_5gs_mobile_identity(Guti5GSMobileIdentity_t *guti, uint8_t *buffer)
{
  int decoded = 0;
  uint16_t temp;
  guti->spare = (*(buffer + decoded) >> 4) & 0xf;

  /*
   * For the 5G-GUTI, bits 5 to 8 of octet 3 are coded as "1111"
   */
  if (guti->spare != 0xf) {
    return (TLV_ENCODE_VALUE_DOESNT_MATCH);
  }


  guti->oddeven = (*(buffer + decoded) >> 3) & 0x1;

  /*
   * For the 5G-GUTI, bits 4 of octet 3 are coded as "0"
   */
  if (guti->oddeven != 0) {
    return (TLV_ENCODE_VALUE_DOESNT_MATCH);
  }
  guti->typeofidentity = *(buffer + decoded) & 0x7;

  if (guti->typeofidentity != FGS_MOBILE_IDENTITY_5G_GUTI) {
    return (TLV_ENCODE_VALUE_DOESNT_MATCH);
  }

  decoded++;
  guti->mccdigit2 = (*(buffer + decoded) >> 4) & 0xf;
  guti->mccdigit1 = *(buffer + decoded) & 0xf;
  decoded++;
  guti->mncdigit3 = (*(buffer + decoded) >> 4) & 0xf;
  guti->mccdigit3 = *(buffer + decoded) & 0xf;
  decoded++;
  guti->mncdigit2 = (*(buffer + decoded) >> 4) & 0xf;
  guti->mncdigit1 = *(buffer + decoded) & 0xf;
  decoded++;


  guti->amfregionid = *(buffer+decoded);
  decoded++;
  IES_DECODE_U16(buffer, decoded, temp);
  guti->amfsetid = temp>>3;
  guti->amfpointer = temp&0x3f;

  IES_DECODE_U32(buffer, decoded, guti->tmsi);
  return decoded;
}


static int encode_guti_5gs_mobile_identity(Guti5GSMobileIdentity_t *guti, uint8_t *buffer)
{
  uint32_t encoded = 0;
  uint16_t temp;
  *(buffer + encoded) = 0xf0 | ((guti->oddeven & 0x1) << 3) |
                        (guti->typeofidentity & 0x7);
  encoded++;
  *(buffer + encoded) = 0x00 | ((guti->mccdigit2 & 0xf) << 4) |
                        (guti->mccdigit1 & 0xf);
  encoded++;
  *(buffer + encoded) = 0x00 | ((guti->mncdigit3 & 0xf) << 4) |
                        (guti->mccdigit3 & 0xf);
  encoded++;
  *(buffer + encoded) = 0x00 | ((guti->mncdigit2 & 0xf) << 4) |
                        (guti->mncdigit1 & 0xf);
  encoded++;

  *(buffer + encoded) = guti->amfregionid;
  encoded++;

  temp = 0x00 | ((guti->amfsetid) << 6) | (guti->amfpointer & 0x3f);

  IES_ENCODE_U16(buffer, encoded, temp);
  IES_ENCODE_U32(buffer, encoded, guti->tmsi);
  return encoded;
}


static int encode_suci_5gs_mobile_identity(Suci5GSMobileIdentity_t *suci, uint8_t *buffer)
{
  uint32_t encoded = 0;
  *(buffer + encoded) = 0x00 | (suci->supiformat << 4) | (suci->typeofidentity);
  encoded++;
  *(buffer + encoded) = 0x00 | ((suci->mccdigit2 & 0xf) << 4) |
                        (suci->mccdigit1 & 0xf);
  encoded++;
  *(buffer + encoded) = 0x00 | ((suci->mncdigit3 & 0xf) << 4) |
                        (suci->mccdigit3 & 0xf);
  encoded++;
  *(buffer + encoded) = 0x00 | ((suci->mncdigit2 & 0xf) << 4) |
                        (suci->mncdigit1 & 0xf);
  encoded++;

  *(buffer + encoded) = 0x00 | ((suci->routingindicatordigit2 & 0xf) << 4) |
                        (suci->routingindicatordigit1 & 0xf);
  encoded++;

  *(buffer + encoded) = 0x00 | ((suci->routingindicatordigit4 & 0xf) << 4) |
                        (suci->routingindicatordigit3 & 0xf);
  encoded++;

  *(buffer + encoded) = 0x00 | (suci->protectionschemeId & 0xf);
  encoded++;

  *(buffer + encoded) = suci->homenetworkpki;
  encoded++;

  IES_ENCODE_U32(buffer, encoded, suci->schemeoutput);

  return encoded;
}

static int encode_imeisv_5gs_mobile_identity(Imeisv5GSMobileIdentity_t *imeisv, uint8_t *buffer)
{
  uint32_t encoded = 0;
  *(buffer + encoded) = 0x00 | (imeisv->digit1 << 4) | (imeisv->oddeven << 3) | (imeisv->typeofidentity);
  encoded++;

  *(buffer + encoded) = 0x00 | (imeisv->digitp1 << 4) | (imeisv->digitp);
  encoded++;

  return encoded;
}

