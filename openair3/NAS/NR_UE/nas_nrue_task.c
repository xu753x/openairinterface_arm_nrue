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

#define NAS_BUILT_IN_UE 1 //QUES: #undef
#define __LITTLE_ENDIAN_BITFIELD 1

#include "utils.h"
# include "assertions.h"
# include "intertask_interface.h"
# include "nas_nrue_task.h"
# include "common/utils/LOG/log.h"

# include "user_defs.h"
# include "user_api.h"
# include "nas_parser.h"
# include "nas_proc.h"
# include "msc.h"
# include "memory.h"

#include "nas_user.h"

// FIXME make command line option for NAS_UE_AUTOSTART
# define NAS_UE_AUTOSTART 1

// FIXME review these externs
extern unsigned char NB_eNB_INST;
extern uint16_t NB_UE_INST;

void *nas_ue_task(void *args_p)
{
  int                   nb_events;
  struct epoll_event   *events;
  MessageDef           *msg_p;
  instance_t            instance;
  unsigned int          Mod_id;
  int                   result;
  nas_user_container_t *users=args_p;
  UENAS_msg * nrue_msg;

  itti_mark_task_ready (TASK_NAS_UE);
  MSC_START_USE();
  
  while(1) {
    // Wait for a message or an event
    itti_receive_msg (TASK_NAS_UE, &msg_p);

    if (msg_p != NULL) {
      instance = ITTI_MSG_INSTANCE (msg_p);
      Mod_id = instance - NB_eNB_INST;
      if (instance == INSTANCE_DEFAULT) {
        printf("%s:%d: FATAL: instance is INSTANCE_DEFAULT, should not happen.\n",
               __FILE__, __LINE__);
        exit_fun("exit... \n");
      }

      switch (ITTI_MSG_ID(msg_p)) {
      case INITIALIZE_MESSAGE:
        LOG_I(NAS, "[UE %d] Received %s\n", Mod_id,  ITTI_MSG_NAME (msg_p));

        break;

      case TERMINATE_MESSAGE:
        itti_exit_task ();
        break;

      case MESSAGE_TEST:
        LOG_I(NAS, "[UE %d] Received %s\n", Mod_id,  ITTI_MSG_NAME (msg_p));
        break;

      case NAS_CELL_SELECTION_CNF:  //CUC：NAS_CELL_SELECTION_CNF √
        LOG_I(NAS, "[UE %d] Received %s: errCode %u, cellID %u, tac %u\n", Mod_id,  ITTI_MSG_NAME (msg_p),
              NAS_CELL_SELECTION_CNF (msg_p).errCode, NAS_CELL_SELECTION_CNF (msg_p).cellID, NAS_CELL_SELECTION_CNF (msg_p).tac);
        as_stmsi_t s_tmsi={0,0};
        plmn_t plmnID={0, 0, 0, 0};
        Byte_t  data = 0;
        nas_itti_nas_establish_req(0, AS_TYPE_ORIGINATING_SIGNAL, s_tmsi, plmnID, &data, 0, 0);
        break;

      case NAS_CELL_SELECTION_IND:
        LOG_I(NAS, "[UE %d] Received %s: cellID %u, tac %u\n", Mod_id,  ITTI_MSG_NAME (msg_p),
              NAS_CELL_SELECTION_IND (msg_p).cellID, NAS_CELL_SELECTION_IND (msg_p).tac);

        /* TODO not processed by NAS currently */
        break;

      case NAS_PAGING_IND:
        LOG_I(NAS, "[UE %d] Received %s: cause %u\n", Mod_id,  ITTI_MSG_NAME (msg_p),
              NAS_PAGING_IND (msg_p).cause);

        /* TODO not processed by NAS currently */
        break;

      case NAS_CONN_ESTABLI_CNF:
        LOG_I(NAS, "[UE %d] Received %s: errCode %u, length %u\n", Mod_id,  ITTI_MSG_NAME (msg_p),
              NAS_CONN_ESTABLI_CNF (msg_p).errCode, NAS_CONN_ESTABLI_CNF (msg_p).nasMsg.length);

        break;

      case NAS_CONN_RELEASE_IND:
        LOG_I(NAS, "[UE %d] Received %s: cause %u\n", Mod_id,  ITTI_MSG_NAME (msg_p),
              NAS_CONN_RELEASE_IND (msg_p).cause);

        break;

      case NAS_UPLINK_DATA_CNF:
        LOG_I(NAS, "[UE %d] Received %s: UEid %u, errCode %u\n", Mod_id,  ITTI_MSG_NAME (msg_p),
              NAS_UPLINK_DATA_CNF (msg_p).UEid, NAS_UPLINK_DATA_CNF (msg_p).errCode);

        break;

      case NAS_DOWNLINK_DATA_IND: //CUC：NAS_DOWNLINK_DATA_IND √
        LOG_I(NAS, "[UE %d] Received %s: UEid %u, length %u\n", Mod_id,  ITTI_MSG_NAME (msg_p),
              NAS_DOWNLINK_DATA_IND (msg_p).UEid, NAS_DOWNLINK_DATA_IND (msg_p).nasMsg.length);
        nr_nas_proc_dl_transfer_ind (nrue_msg, NAS_DOWNLINK_DATA_IND(msg_p).nasMsg.data, NAS_DOWNLINK_DATA_IND(msg_p).nasMsg.length); //handle dl info NAS mesaages.
        break;

      default:
        LOG_E(NAS, "[UE %d] Received unexpected message %s\n", Mod_id,  ITTI_MSG_NAME (msg_p));
        break;
      }

      result = itti_free (ITTI_MSG_ORIGIN_ID(msg_p), msg_p);
      AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
      msg_p = NULL;
    }


  }

  free(users);
  return NULL;
}

void nr_nas_proc_dl_transfer_ind (UENAS_msg *msg,  Byte_t *data, uint32_t len) { //QUES: 解出的msg干什么
  uint8_t *buffer;
  UENAS_msg *msg1;
  uint32_t len1;
  nr_user_nas_t UErrc= {0};//QUES:user
  int size;
  decodeNasMsg(msg,data,len);
  switch (msg->header.message_type) {
    case IDENTITY_REQUEST: {     //send identityResponse in NAS_UPLINK_DATA_REQ
      msg1->header.protocol_discriminator=0;
      msg1->header.security_header_type=0;
      len1 += sizeof(uint8_t);
      msg1->header.message_type = IDENTITY_RESPONSE;
      len1 += sizeof(uint8_t);
      len1 += identityResponse((void **)&msg1->identity_response, &UErrc);
      size = encodeNasMsg(msg1, buffer, len1);
      nas_itti_ul_data_req(0,buffer,size,0); 
      break;
      }
      
    case AUTHENTICATION_REQUEST: { //send authenticationResponse
      msg1->header.protocol_discriminator=0;
      msg1->header.security_header_type=0;
      len1 += sizeof(uint8_t);
      msg1->header.message_type = AUTHENTICATION_RESPONSE;
      len1 += sizeof(uint8_t);
      len1 += authenticationResponse((void **)&msg1->identity_response, &UErrc);
      size = encodeNasMsg(msg1, buffer, len1);
      nas_itti_ul_data_req(0,buffer,size,0);
      break;
      }
      
  }
}

int decodeNasMsg(UENAS_msg *msg, uint8_t *buffer, uint32_t len) {
  int header_result;
  int decode_result=0;

  /* First decode the EMM message header */
  header_result = _emm_msg_decode_header(&msg->header, buffer, len);

  if (header_result < 0) {
    LOG_TRACE(ERROR, "NR_UE   - Failed to decode EMM message header "
              "(%d)", header_result);
    LOG_FUNC_RETURN(header_result);
  }

  buffer += header_result;
  len -= header_result;
  LOG_TRACE(INFO, "NR_UE   - Message Type 0x%02x", msg->header.message_type);

  switch(msg->header.message_type) { 

  }

  LOG_FUNC_RETURN (header_result + decode_result);
}

int encodeNasMsg(UENAS_msg *msg, uint8_t *buffer, uint32_t len) { //QUES:UENAS_msg *msg
  int header_result;
  int encode_result=0;

  /* First encode the EMM message header */
  header_result = _emm_msg_encode_header(&msg->header, buffer, len);

  if (header_result < 0) {
    LOG_TRACE(ERROR, "NR_UE   - Failed to encode EMM message header "
              "(%d)", header_result);
    LOG_FUNC_RETURN(header_result);
  }

  buffer += header_result;
  len -= header_result;

  switch(msg->header.message_type) { 
    case IDENTITY_RESPONSE: {
      encode_result = encode_identity_response(&msg->identity_response, buffer, len);
      break;
    }
    case AUTHENTICATION_RESPONSE: {
      encode_result = encode_authentication_response(&msg->authentication_response, buffer, len);
      break;
    }
  }
  LOG_FUNC_RETURN (header_result + encode_result);
}

static int _emm_msg_decode_header(emm_msg_header_t *header,
                                  const uint8_t *buffer, uint32_t len) {
  int size = 0;

  /* Check the buffer length */

  /* Decode the security header type and the protocol discriminator */
  DECODE_U8(buffer + size, *(uint8_t *)(header), size);
  /* Decode the message type */
  DECODE_U8(buffer + size, header->message_type, size);

  /* Check the protocol discriminator */

  return (size);
}

static int _emm_msg_encode_header(const emm_msg_header_t *header,
                                  uint8_t *buffer, uint32_t len) {
  int size = 0;

  /* Check the buffer length */

  /* Check the protocol discriminator */

  /* Encode the security header type and the protocol discriminator */
  ENCODE_U8(buffer + size, *(uint8_t *)(header), size);
  /* Encode the message type */
  ENCODE_U8(buffer + size, header->message_type, size);
  return (size);
}

int encode_IdentityresponseIMSI(IdentityresponseIMSI_t *identity_response, uint8_t *buffer, uint32_t len)
{
  // int encoded = 0;
  // int encode_result = 0;

  // /* Checking IEI and pointer */
  // CHECK_PDU_POINTER_AND_LENGTH_ENCODER(buffer, IDENTITY_RESPONSE_MINIMUM_LENGTH, len);

  // if ((encode_result =
  //        encode_mobile_identity(&identity_response->mobileidentity, 0, buffer +
  //                               encoded, len - encoded)) < 0)        //Return in case of error
  //   return encode_result;
  // else
  //   encoded += encode_result;

  // return encoded;
}

int encode_authenticationresponse(authenticationresponse_t *authentication_response, uint8_t *buffer, uint32_t len)
{
  // int encoded = 0;
  // int encode_result = 0;

  // /* Checking IEI and pointer */
  // CHECK_PDU_POINTER_AND_LENGTH_ENCODER(buffer, AUTHENTICATION_RESPONSE_MINIMUM_LENGTH, len);

  // if ((encode_result =
  //        encode_authentication_response_parameter(&authentication_response->authenticationresponseparameter,
  //            0, buffer + encoded, len - encoded)) < 0)        //Return in case of error
  //   return encode_result;
  // else
  //   encoded += encode_result;

  // return encoded;
}