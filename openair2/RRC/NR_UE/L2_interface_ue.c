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

/* \file l2_interface_ue.c
 * \brief layer 2 interface, used to support different RRC sublayer
 * \author R. Knopp, K.H. HSU
 * \date 2018
 * \version 0.1
 * \company Eurecom / NTUST
 * \email: knopp@eurecom.fr, kai-hsiang.hsu@eurecom.fr
 * \note
 * \warning
 */

#include "rrc_defs.h"
#include "rrc_proto.h"
#include "assertions.h"
#include "rrc_vars.h"

typedef uint32_t channel_t;

int8_t
nr_mac_rrc_data_ind_ue(
    const module_id_t     module_id,
    const int             CC_id,
    const uint8_t         gNB_index,
    const frame_t         frame,
    const sub_frame_t     sub_frame,
    const rnti_t          rnti,
    const channel_t       channel,
    const uint8_t*        pduP,
    const sdu_size_t      pdu_len){
    sdu_size_t      sdu_size = 0;
    protocol_ctxt_t ctxt;

    switch(channel){
        case NR_BCCH_BCH:
            AssertFatal( nr_rrc_ue_decode_NR_BCCH_BCH_Message( module_id, gNB_index, (uint8_t*)pduP, pdu_len) == 0, "UE decode BCCH-BCH error!\n");
            break;
        case CCCH:
             if (pdu_len>0) {
                LOG_T(RRC,"[UE %d] Received SDU for CCCH on SRB %ld from gNB %d\n",module_id,channel & RAB_OFFSET,gNB_index);
                {
                  MessageDef *message_p;
                  int msg_sdu_size = CCCH_SDU_SIZE;

                  if (pdu_len > msg_sdu_size) {
                    LOG_E(RRC, "SDU larger than CCCH SDU buffer size (%d, %d)", sdu_size, msg_sdu_size);
                    sdu_size = msg_sdu_size;
                  } else {
                    sdu_size =  pdu_len;
                  }

                  message_p = itti_alloc_new_message (TASK_MAC_UE, NR_RRC_MAC_CCCH_DATA_IND); 
                  memset (NR_RRC_MAC_CCCH_DATA_IND (message_p).sdu, 0, CCCH_SDU_SIZE);
                  memcpy (NR_RRC_MAC_CCCH_DATA_IND (message_p).sdu, pduP, sdu_size);
                  NR_RRC_MAC_CCCH_DATA_IND (message_p).frame     = frame; //frameP
                  NR_RRC_MAC_CCCH_DATA_IND (message_p).sub_frame = sub_frame; //sub_frameP
                  NR_RRC_MAC_CCCH_DATA_IND (message_p).sdu_size  = sdu_size;
                  NR_RRC_MAC_CCCH_DATA_IND (message_p).gnb_index = gNB_index;
                  NR_RRC_MAC_CCCH_DATA_IND (message_p).rnti      = rnti;  //rntiP
                  itti_send_msg_to_task (TASK_RRC_NRUE, GNB_MODULE_ID_TO_INSTANCE( module_id ), message_p);
                }
            }
        default:
            break;
    }


    return(0);

}

int8_t mac_rrc_nr_data_req_ue(  
  const module_id_t Mod_idP,
  const int         CC_id,
  const frame_t     frameP,
  const rb_id_t     Srb_id,
  const uint8_t     Nb_tb,
  uint8_t    *const buffer_pP,
  const uint8_t     gNB_index,
  const uint8_t     mbsfn_sync_area
){

  // todo
  if( (NR_UE_rrc_inst[Mod_idP].Srb0[gNB_index].Tx_buffer.payload_size > 0) ) {
    
  //     MessageDef *message_p;
  //     int ccch_size = NR_UE_rrc_inst[Mod_idP].Srb0[gNB_index].Tx_buffer.payload_size;
  //     int sdu_size = sizeof(RRC_MAC_CCCH_DATA_REQ (message_p).sdu);

  //     if (ccch_size > sdu_size) {
  //       LOG_E(RRC, "SDU larger than CCCH SDU buffer size (%d, %d)", ccch_size, sdu_size);
  //       ccch_size = sdu_size;
  //     }

  //     message_p = itti_alloc_new_message (TASK_RRC_UE, RRC_MAC_CCCH_DATA_REQ);
  //     RRC_MAC_CCCH_DATA_REQ (message_p).frame = frameP;
  //     RRC_MAC_CCCH_DATA_REQ (message_p).sdu_size = ccch_size;
  //     memset (RRC_MAC_CCCH_DATA_REQ (message_p).sdu, 0, CCCH_SDU_SIZE);
  //     memcpy (RRC_MAC_CCCH_DATA_REQ (message_p).sdu, NR_UE_rrc_inst[Mod_idP].Srb0[gNB_index].Tx_buffer.Payload, ccch_size);
  //     RRC_MAC_CCCH_DATA_REQ (message_p).enb_index = gNB_index;

  //     itti_send_msg_to_task (TASK_MAC_UE, UE_MODULE_ID_TO_INSTANCE(Mod_idP), message_p);
  //   }
    memset (buffer_pP, 0,sizeof(buffer_pP));
    memcpy(&buffer_pP[0],&NR_UE_rrc_inst[Mod_idP].Srb0[gNB_index].Tx_buffer.Payload[0],NR_UE_rrc_inst[Mod_idP].Srb0[gNB_index].Tx_buffer.payload_size);
    uint8_t Ret_size=NR_UE_rrc_inst[Mod_idP].Srb0[gNB_index].Tx_buffer.payload_size;
    //   NR_UE_rrc_inst[Mod_id].Srb0[eNB_index].Tx_buffer.payload_size=0;
    NR_UE_rrc_inst[Mod_idP].Info[gNB_index].T300_active = 1;
    NR_UE_rrc_inst[Mod_idP].Info[gNB_index].T300_cnt = 0;
    //      msg("[RRC][UE %d] Sending rach\n",Mod_id);
    return(Ret_size);
  } else {
    return 0;
  }

  return 0;
}

uint8_t
rrc_data_req_ue(
  const protocol_ctxt_t   *const ctxt_pP,
  const rb_id_t                  rb_idP,
  const mui_t                    muiP,
  const confirm_t                confirmP,
  const sdu_size_t               sdu_sizeP,
  uint8_t                 *const buffer_pP,
  const pdcp_transmission_mode_t modeP
)
{
    MessageDef *message_p;
    // Uses a new buffer to avoid issue with PDCP buffer content that could be changed by PDCP (asynchronous message handling).
    uint8_t *message_buffer;
    message_buffer = itti_malloc (
                       TASK_RRC_UE,
                       TASK_PDCP_UE,
                       sdu_sizeP);
    memcpy (message_buffer, buffer_pP, sdu_sizeP);
    message_p = itti_alloc_new_message ( TASK_RRC_UE, RRC_DCCH_DATA_REQ);
    RRC_DCCH_DATA_REQ (message_p).frame     = ctxt_pP->frame;
    RRC_DCCH_DATA_REQ (message_p).enb_flag  = ctxt_pP->enb_flag;
    RRC_DCCH_DATA_REQ (message_p).rb_id     = rb_idP;
    RRC_DCCH_DATA_REQ (message_p).muip      = muiP;
    RRC_DCCH_DATA_REQ (message_p).confirmp  = confirmP;
    RRC_DCCH_DATA_REQ (message_p).sdu_size  = sdu_sizeP;
    RRC_DCCH_DATA_REQ (message_p).sdu_p     = message_buffer;
    RRC_DCCH_DATA_REQ (message_p).mode      = modeP;
    RRC_DCCH_DATA_REQ (message_p).module_id = ctxt_pP->module_id;
    RRC_DCCH_DATA_REQ (message_p).rnti      = ctxt_pP->rnti;
    RRC_DCCH_DATA_REQ (message_p).eNB_index = ctxt_pP->eNB_index;
    itti_send_msg_to_task (
      TASK_PDCP_UE,
      ctxt_pP->instance,
      message_p);
    return TRUE; // TODO should be changed to a CNF message later, currently RRC lite does not used the returned value anyway.

}

