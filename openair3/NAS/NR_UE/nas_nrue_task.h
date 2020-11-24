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

#ifndef NAS_NRUE_TASK_H_
#define NAS_NRUE_TASK_H_

#include "openairinterface5g_limits.h"

//CUC：add nas_nrue_task.h √
#include "platform_types.h"
#include "nas_ue_task.h"
#include "nas_itti_messaging.h"
#include "nas_log.h"
#include "TLVDecoder.h"
#include "TLVEncoder.h"
#include "NR_NAS_defs.h"
#include "nr_nas_msg_sim.h"

# define REGISTRATION_ACCEPT        0b01000010
# define REGISTRATION_COMPLETE      0b01000011
typedef struct  __attribute__((packed)) {
  Extendedprotocoldiscriminator_t epd:8;
  Security_header_t sh:8;
  SGSmobilitymanagementmessages_t mt:8;
} securityModeComplete_t;

typedef struct  __attribute__((packed)) {
  unsigned int len:8;
  unsigned int allowed:4;
  unsigned int value:4;
} SGSregistrationresult;

typedef struct  __attribute__((packed)) {
  Extendedprotocoldiscriminator_t epd:8;
  Security_header_t sh:8;
  SGSmobilitymanagementmessages_t mt:8;
  SGSregistrationresult rr;
} registrationaccept_t;

typedef struct  __attribute__((packed)) {
  Extendedprotocoldiscriminator_t epd:8;
  Security_header_t sh:8;
  SGSmobilitymanagementmessages_t mt:8;
} registrationcomplete_t;

typedef union {
  mm_msg_header_t header;
  authenticationrequestHeader_t authentication_request;
  authenticationresponse_t authentication_response;
  Identityrequest_t identity_request;
  IdentityresponseIMSI_t identity_response;
  securityModeCommand_t securitymode_command;
  securityModeComplete_t securitymode_complete;
  registrationaccept_t registration_accept;
  registrationcomplete_t registration_complete;
} UENAS_msg;

void *nas_nrue_task(void *args_p);
void nr_nas_proc_dl_transfer_ind (UENAS_msg *msg, Byte_t *data, uint32_t len);
int decodeNasMsg(UENAS_msg *msg, uint8_t *buffer, uint32_t len);
int encodeNasMsg(UENAS_msg *msg, uint8_t *buffer, uint32_t len);
int encode_authentication_response5g(authenticationresponse_t *authentication_response, uint8_t *buffer, uint32_t len);
int encode_security_mode_complete5g(securityModeComplete_t *securitymodecomplete, uint8_t *buffer, uint32_t len);
int encode_registration_complete5g(registrationcomplete_t *registrationcomplete, uint8_t *buffer, uint32_t len);
int authenticationResponse5g(authenticationresponse_t *msg);
int securityModeComplete5g(securityModeComplete_t *msg);
int registrationComplete5g(registrationcomplete_t *msg);

int string2ByteArray(char* input,uint8_t* output); //CUC:test
void tesths(void);

#endif /* NAS_TASK_H_ */
