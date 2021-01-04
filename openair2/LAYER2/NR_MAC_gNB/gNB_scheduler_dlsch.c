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

/*! \file       gNB_scheduler_dlsch.c
 * \brief       procedures related to gNB for the DLSCH transport channel
 * \author      Guido Casati
 * \date        2019
 * \email:      guido.casati@iis.fraunhofe.de
 * \version     1.0
 * @ingroup     _mac

 */

/*PHY*/
#include "PHY/CODING/coding_defs.h"
#include "PHY/defs_nr_common.h"
#include "common/utils/nr/nr_common.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
/*MAC*/
#include "NR_MAC_COMMON/nr_mac.h"
#include "NR_MAC_gNB/nr_mac_gNB.h"
#include "NR_MAC_COMMON/nr_mac_extern.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"

/*NFAPI*/
#include "nfapi_nr_interface.h"
/*TAG*/
#include "NR_TAG-Id.h"

/*Softmodem params*/
#include "executables/softmodem-common.h"

////////////////////////////////////////////////////////
/////* DLSCH MAC PDU generation (6.1.2 TS 38.321) */////
////////////////////////////////////////////////////////
#define OCTET 8
#define HALFWORD 16
#define WORD 32
//#define SIZE_OF_POINTER sizeof (void *)

int nr_generate_dlsch_pdu(module_id_t module_idP,
                          NR_UE_sched_ctrl_t *ue_sched_ctl,
                          unsigned char *sdus_payload,
                          unsigned char *mac_pdu,
                          unsigned char num_sdus,
                          unsigned short *sdu_lengths,
                          unsigned char *sdu_lcids,
                          unsigned char drx_cmd,
                          unsigned char *ue_cont_res_id,
                          unsigned short post_padding) {
  gNB_MAC_INST *gNB = RC.nrmac[module_idP];
  NR_MAC_SUBHEADER_FIXED *mac_pdu_ptr = (NR_MAC_SUBHEADER_FIXED *) mac_pdu;
  unsigned char *dlsch_buffer_ptr = sdus_payload;
  uint8_t last_size = 0;
  int offset = 0, mac_ce_size, i, timing_advance_cmd, tag_id = 0;
  // MAC CEs
  uint8_t mac_header_control_elements[16], *ce_ptr;
  ce_ptr = &mac_header_control_elements[0];

  // 1) Compute MAC CE and related subheaders

  // DRX command subheader (MAC CE size 0)
  if (drx_cmd != 255) {
    mac_pdu_ptr->R = 0;
    mac_pdu_ptr->LCID = DL_SCH_LCID_DRX;
    //last_size = 1;
    mac_pdu_ptr++;
  }

  // Timing Advance subheader
  /* This was done only when timing_advance_cmd != 31
  // now TA is always send when ta_timer resets regardless of its value
  // this is done to avoid issues with the timeAlignmentTimer which is
  // supposed to monitor if the UE received TA or not */
  if (ue_sched_ctl->ta_apply) {
    mac_pdu_ptr->R = 0;
    mac_pdu_ptr->LCID = DL_SCH_LCID_TA_COMMAND;
    //last_size = 1;
    mac_pdu_ptr++;
    // TA MAC CE (1 octet)
    timing_advance_cmd = ue_sched_ctl->ta_update;
    AssertFatal(timing_advance_cmd < 64, "timing_advance_cmd %d > 63\n", timing_advance_cmd);
    ((NR_MAC_CE_TA *) ce_ptr)->TA_COMMAND = timing_advance_cmd;    //(timing_advance_cmd+31)&0x3f;

    if (gNB->tag->tag_Id != 0) {
      tag_id = gNB->tag->tag_Id;
      ((NR_MAC_CE_TA *) ce_ptr)->TAGID = tag_id;
    }

    LOG_D(MAC, "NR MAC CE timing advance command = %d (%d) TAG ID = %d\n", timing_advance_cmd, ((NR_MAC_CE_TA *) ce_ptr)->TA_COMMAND, tag_id);
    mac_ce_size = sizeof(NR_MAC_CE_TA);
    // Copying  bytes for MAC CEs to the mac pdu pointer
    memcpy((void *) mac_pdu_ptr, (void *) ce_ptr, mac_ce_size);
    ce_ptr += mac_ce_size;
    mac_pdu_ptr += (unsigned char) mac_ce_size;


  }

  // Contention resolution fixed subheader and MAC CE
  if (ue_cont_res_id) {
    mac_pdu_ptr->R = 0;
    mac_pdu_ptr->LCID = DL_SCH_LCID_CON_RES_ID;
    mac_pdu_ptr++;
    //last_size = 1;
    // contention resolution identity MAC ce has a fixed 48 bit size
    // this contains the UL CCCH SDU. If UL CCCH SDU is longer than 48 bits,
    // it contains the first 48 bits of the UL CCCH SDU
    LOG_T(MAC, "[gNB ][RAPROC] Generate contention resolution msg: %x.%x.%x.%x.%x.%x\n",
          ue_cont_res_id[0], ue_cont_res_id[1], ue_cont_res_id[2],
          ue_cont_res_id[3], ue_cont_res_id[4], ue_cont_res_id[5]);
    // Copying bytes (6 octects) to CEs pointer
    mac_ce_size = 6;
    memcpy(ce_ptr, ue_cont_res_id, mac_ce_size);
    // Copying bytes for MAC CEs to mac pdu pointer
    memcpy((void *) mac_pdu_ptr, (void *) ce_ptr, mac_ce_size);
    ce_ptr += mac_ce_size;
    mac_pdu_ptr += (unsigned char) mac_ce_size;
  }

  //TS 38.321 Sec 6.1.3.15 TCI State indication for UE Specific PDCCH MAC CE SubPDU generation
  if (ue_sched_ctl->UE_mac_ce_ctrl.pdcch_state_ind.is_scheduled) {
    //filling subheader
    mac_pdu_ptr->R = 0;
    mac_pdu_ptr->LCID = DL_SCH_LCID_TCI_STATE_IND_UE_SPEC_PDCCH;
    mac_pdu_ptr++;
    //Creating the instance of CE structure
    NR_TCI_PDCCH  nr_UESpec_TCI_StateInd_PDCCH;
    //filling the CE structre
    nr_UESpec_TCI_StateInd_PDCCH.CoresetId1 = ((ue_sched_ctl->UE_mac_ce_ctrl.pdcch_state_ind.coresetId) & 0xF) >> 1; //extracting MSB 3 bits from LS nibble
    nr_UESpec_TCI_StateInd_PDCCH.ServingCellId = (ue_sched_ctl->UE_mac_ce_ctrl.pdcch_state_ind.servingCellId) & 0x1F; //extracting LSB 5 Bits
    nr_UESpec_TCI_StateInd_PDCCH.TciStateId = (ue_sched_ctl->UE_mac_ce_ctrl.pdcch_state_ind.tciStateId) & 0x7F; //extracting LSB 7 bits
    nr_UESpec_TCI_StateInd_PDCCH.CoresetId2 = (ue_sched_ctl->UE_mac_ce_ctrl.pdcch_state_ind.coresetId) & 0x1; //extracting LSB 1 bit
    LOG_D(MAC, "NR MAC CE TCI state indication for UE Specific PDCCH = %d \n", nr_UESpec_TCI_StateInd_PDCCH.TciStateId);
    mac_ce_size = sizeof(NR_TCI_PDCCH);
    // Copying  bytes for MAC CEs to the mac pdu pointer
    memcpy((void *) mac_pdu_ptr, (void *)&nr_UESpec_TCI_StateInd_PDCCH, mac_ce_size);
    //incrementing the PDU pointer
    mac_pdu_ptr += (unsigned char) mac_ce_size;
  }

  //TS 38.321 Sec 6.1.3.16, SP CSI reporting on PUCCH Activation/Deactivation MAC CE
  if (ue_sched_ctl->UE_mac_ce_ctrl.SP_CSI_reporting_pucch.is_scheduled) {
    //filling the subheader
    mac_pdu_ptr->R = 0;
    mac_pdu_ptr->LCID = DL_SCH_LCID_SP_CSI_REP_PUCCH_ACT;
    mac_pdu_ptr++;
    //creating the instance of CE structure
    NR_PUCCH_CSI_REPORTING nr_PUCCH_CSI_reportingActDeact;
    //filling the CE structure
    nr_PUCCH_CSI_reportingActDeact.BWP_Id = (ue_sched_ctl->UE_mac_ce_ctrl.SP_CSI_reporting_pucch.bwpId) & 0x3; //extracting LSB 2 bibs
    nr_PUCCH_CSI_reportingActDeact.ServingCellId = (ue_sched_ctl->UE_mac_ce_ctrl.SP_CSI_reporting_pucch.servingCellId) & 0x1F; //extracting LSB 5 bits
    nr_PUCCH_CSI_reportingActDeact.S0 = ue_sched_ctl->UE_mac_ce_ctrl.SP_CSI_reporting_pucch.s0tos3_actDeact[0];
    nr_PUCCH_CSI_reportingActDeact.S1 = ue_sched_ctl->UE_mac_ce_ctrl.SP_CSI_reporting_pucch.s0tos3_actDeact[1];
    nr_PUCCH_CSI_reportingActDeact.S2 = ue_sched_ctl->UE_mac_ce_ctrl.SP_CSI_reporting_pucch.s0tos3_actDeact[2];
    nr_PUCCH_CSI_reportingActDeact.S3 = ue_sched_ctl->UE_mac_ce_ctrl.SP_CSI_reporting_pucch.s0tos3_actDeact[3];
    nr_PUCCH_CSI_reportingActDeact.R2 = 0;
    mac_ce_size = sizeof(NR_PUCCH_CSI_REPORTING);
    // Copying MAC CE data to the mac pdu pointer
    memcpy((void *) mac_pdu_ptr, (void *)&nr_PUCCH_CSI_reportingActDeact, mac_ce_size);
    //incrementing the PDU pointer
    mac_pdu_ptr += (unsigned char) mac_ce_size;
  }

  //TS 38.321 Sec 6.1.3.14, TCI State activation/deactivation for UE Specific PDSCH MAC CE
  if (ue_sched_ctl->UE_mac_ce_ctrl.pdsch_TCI_States_ActDeact.is_scheduled) {
    //Computing the number of octects to be allocated for Flexible array member
    //of MAC CE structure
    uint8_t num_octects = (ue_sched_ctl->UE_mac_ce_ctrl.pdsch_TCI_States_ActDeact.highestTciStateActivated) / 8 + 1; //Calculating the number of octects for allocating the memory
    //filling the subheader
    ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->R = 0;
    ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->F = 0;
    ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->LCID = DL_SCH_LCID_TCI_STATE_ACT_UE_SPEC_PDSCH;
    ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->L = sizeof(NR_TCI_PDSCH_APERIODIC_CSI) + num_octects * sizeof(uint8_t);
    last_size = 2;
    //Incrementing the PDU pointer
    mac_pdu_ptr += last_size;
    //allocating memory for CE Structure
    NR_TCI_PDSCH_APERIODIC_CSI *nr_UESpec_TCI_StateInd_PDSCH = (NR_TCI_PDSCH_APERIODIC_CSI *)malloc(sizeof(NR_TCI_PDSCH_APERIODIC_CSI) + num_octects * sizeof(uint8_t));
    //initializing to zero
    memset((void *)nr_UESpec_TCI_StateInd_PDSCH, 0, sizeof(NR_TCI_PDSCH_APERIODIC_CSI) + num_octects * sizeof(uint8_t));
    //filling the CE Structure
    nr_UESpec_TCI_StateInd_PDSCH->BWP_Id = (ue_sched_ctl->UE_mac_ce_ctrl.pdsch_TCI_States_ActDeact.bwpId) & 0x3; //extracting LSB 2 Bits
    nr_UESpec_TCI_StateInd_PDSCH->ServingCellId = (ue_sched_ctl->UE_mac_ce_ctrl.pdsch_TCI_States_ActDeact.servingCellId) & 0x1F; //extracting LSB 5 bits

    for(i = 0; i < (num_octects * 8); i++) {
      if(ue_sched_ctl->UE_mac_ce_ctrl.pdsch_TCI_States_ActDeact.tciStateActDeact[i])
        nr_UESpec_TCI_StateInd_PDSCH->T[i / 8] = nr_UESpec_TCI_StateInd_PDSCH->T[i / 8] | (1 << (i % 8));
    }

    mac_ce_size = sizeof(NR_TCI_PDSCH_APERIODIC_CSI) + num_octects * sizeof(uint8_t);
    //Copying  bytes for MAC CEs to the mac pdu pointer
    memcpy((void *) mac_pdu_ptr, (void *)nr_UESpec_TCI_StateInd_PDSCH, mac_ce_size);
    //incrementing the mac pdu pointer
    mac_pdu_ptr += (unsigned char) mac_ce_size;
    //freeing the allocated memory
    free(nr_UESpec_TCI_StateInd_PDSCH);
  }

  //TS38.321 Sec 6.1.3.13 Aperiodic CSI Trigger State Subselection MAC CE
  if (ue_sched_ctl->UE_mac_ce_ctrl.aperi_CSI_trigger.is_scheduled) {
    //Computing the number of octects to be allocated for Flexible array member
    //of MAC CE structure
    uint8_t num_octects = (ue_sched_ctl->UE_mac_ce_ctrl.aperi_CSI_trigger.highestTriggerStateSelected) / 8 + 1; //Calculating the number of octects for allocating the memory
    //filling the subheader
    ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->R = 0;
    ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->F = 0;
    ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->LCID = DL_SCH_LCID_APERIODIC_CSI_TRI_STATE_SUBSEL;
    ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->L = sizeof(NR_TCI_PDSCH_APERIODIC_CSI) + num_octects * sizeof(uint8_t);
    last_size = 2;
    //Incrementing the PDU pointer
    mac_pdu_ptr += last_size;
    //allocating memory for CE structure
    NR_TCI_PDSCH_APERIODIC_CSI *nr_Aperiodic_CSI_Trigger = (NR_TCI_PDSCH_APERIODIC_CSI *)malloc(sizeof(NR_TCI_PDSCH_APERIODIC_CSI) + num_octects * sizeof(uint8_t));
    //initializing to zero
    memset((void *)nr_Aperiodic_CSI_Trigger, 0, sizeof(NR_TCI_PDSCH_APERIODIC_CSI) + num_octects * sizeof(uint8_t));
    //filling the CE Structure
    nr_Aperiodic_CSI_Trigger->BWP_Id = (ue_sched_ctl->UE_mac_ce_ctrl.aperi_CSI_trigger.bwpId) & 0x3; //extracting LSB 2 bits
    nr_Aperiodic_CSI_Trigger->ServingCellId = (ue_sched_ctl->UE_mac_ce_ctrl.aperi_CSI_trigger.servingCellId) & 0x1F; //extracting LSB 5 bits
    nr_Aperiodic_CSI_Trigger->R = 0;

    for(i = 0; i < (num_octects * 8); i++) {
      if(ue_sched_ctl->UE_mac_ce_ctrl.aperi_CSI_trigger.triggerStateSelection[i])
        nr_Aperiodic_CSI_Trigger->T[i / 8] = nr_Aperiodic_CSI_Trigger->T[i / 8] | (1 << (i % 8));
    }

    mac_ce_size = sizeof(NR_TCI_PDSCH_APERIODIC_CSI) + num_octects * sizeof(uint8_t);
    // Copying  bytes for MAC CEs to the mac pdu pointer
    memcpy((void *) mac_pdu_ptr, (void *)nr_Aperiodic_CSI_Trigger, mac_ce_size);
    //incrementing the mac pdu pointer
    mac_pdu_ptr += (unsigned char) mac_ce_size;
    //freeing the allocated memory
    free(nr_Aperiodic_CSI_Trigger);
  }

  if (ue_sched_ctl->UE_mac_ce_ctrl.sp_zp_csi_rs.is_scheduled) {
    ((NR_MAC_SUBHEADER_FIXED *) mac_pdu_ptr)->R = 0;
    ((NR_MAC_SUBHEADER_FIXED *) mac_pdu_ptr)->LCID = DL_SCH_LCID_SP_ZP_CSI_RS_RES_SET_ACT;
    mac_pdu_ptr++;
    ((NR_MAC_CE_SP_ZP_CSI_RS_RES_SET *) mac_pdu_ptr)->A_D = ue_sched_ctl->UE_mac_ce_ctrl.sp_zp_csi_rs.act_deact;
    ((NR_MAC_CE_SP_ZP_CSI_RS_RES_SET *) mac_pdu_ptr)->CELLID = ue_sched_ctl->UE_mac_ce_ctrl.sp_zp_csi_rs.serv_cell_id & 0x1F; //5 bits
    ((NR_MAC_CE_SP_ZP_CSI_RS_RES_SET *) mac_pdu_ptr)->BWPID = ue_sched_ctl->UE_mac_ce_ctrl.sp_zp_csi_rs.bwpid & 0x3; //2 bits
    ((NR_MAC_CE_SP_ZP_CSI_RS_RES_SET *) mac_pdu_ptr)->CSIRS_RSC_ID = ue_sched_ctl->UE_mac_ce_ctrl.sp_zp_csi_rs.rsc_id & 0xF; //4 bits
    ((NR_MAC_CE_SP_ZP_CSI_RS_RES_SET *) mac_pdu_ptr)->R = 0;
    LOG_D(MAC, "NR MAC CE of ZP CSIRS Serv cell ID = %d BWPID= %d Rsc set ID = %d\n", ue_sched_ctl->UE_mac_ce_ctrl.sp_zp_csi_rs.serv_cell_id, ue_sched_ctl->UE_mac_ce_ctrl.sp_zp_csi_rs.bwpid,
          ue_sched_ctl->UE_mac_ce_ctrl.sp_zp_csi_rs.rsc_id);
    mac_ce_size = sizeof(NR_MAC_CE_SP_ZP_CSI_RS_RES_SET);
    mac_pdu_ptr += (unsigned char) mac_ce_size;
  }

  if (ue_sched_ctl->UE_mac_ce_ctrl.csi_im.is_scheduled) {
    mac_pdu_ptr->R = 0;
    mac_pdu_ptr->LCID = DL_SCH_LCID_SP_CSI_RS_CSI_IM_RES_SET_ACT;
    mac_pdu_ptr++;
    CSI_RS_CSI_IM_ACT_DEACT_MAC_CE csi_rs_im_act_deact_ce;
    csi_rs_im_act_deact_ce.A_D = ue_sched_ctl->UE_mac_ce_ctrl.csi_im.act_deact;
    csi_rs_im_act_deact_ce.SCID = ue_sched_ctl->UE_mac_ce_ctrl.csi_im.serv_cellid & 0x3F;//gNB_PHY -> ssb_pdu.ssb_pdu_rel15.PhysCellId;
    csi_rs_im_act_deact_ce.BWP_ID = ue_sched_ctl->UE_mac_ce_ctrl.csi_im.bwp_id;
    csi_rs_im_act_deact_ce.R1 = 0;
    csi_rs_im_act_deact_ce.IM = ue_sched_ctl->UE_mac_ce_ctrl.csi_im.im;// IF set CSI IM Rsc id will presesent else CSI IM RSC ID is abscent
    csi_rs_im_act_deact_ce.SP_CSI_RSID = ue_sched_ctl->UE_mac_ce_ctrl.csi_im.nzp_csi_rsc_id;

    if ( csi_rs_im_act_deact_ce.IM ) { //is_scheduled if IM is 1 else this field will not present
      csi_rs_im_act_deact_ce.R2 = 0;
      csi_rs_im_act_deact_ce.SP_CSI_IMID = ue_sched_ctl->UE_mac_ce_ctrl.csi_im.csi_im_rsc_id;
      mac_ce_size = sizeof ( csi_rs_im_act_deact_ce ) - sizeof ( csi_rs_im_act_deact_ce.TCI_STATE );
    } else {
      mac_ce_size = sizeof ( csi_rs_im_act_deact_ce ) - sizeof ( csi_rs_im_act_deact_ce.TCI_STATE ) - 1;
    }

    memcpy ((void *) mac_pdu_ptr, (void *) & ( csi_rs_im_act_deact_ce), mac_ce_size);
    mac_pdu_ptr += (unsigned char) mac_ce_size;

    if (csi_rs_im_act_deact_ce.A_D ) { //Following IE is_scheduled only if A/D is 1
      mac_ce_size = sizeof ( struct TCI_S);

      for ( i = 0; i < ue_sched_ctl->UE_mac_ce_ctrl.csi_im.nb_tci_resource_set_id; i++) {
        csi_rs_im_act_deact_ce.TCI_STATE.R = 0;
        csi_rs_im_act_deact_ce.TCI_STATE.TCI_STATE_ID = ue_sched_ctl->UE_mac_ce_ctrl.csi_im.tci_state_id [i] & 0x7F;
        memcpy ((void *) mac_pdu_ptr, (void *) & (csi_rs_im_act_deact_ce.TCI_STATE), mac_ce_size);
        mac_pdu_ptr += (unsigned char) mac_ce_size;
      }
    }
  }

  // 2) Generation of DLSCH MAC subPDUs including subheaders and MAC SDUs
  for (i = 0; i < num_sdus; i++) {

    if (sdu_lengths[i] < 256) {
      ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->R = 0;
      ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->F = 0;
      ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->LCID = sdu_lcids[i];
      ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->L = (unsigned char) sdu_lengths[i];
      last_size = 2;
    } else {
      ((NR_MAC_SUBHEADER_LONG *) mac_pdu_ptr)->R = 0;
      ((NR_MAC_SUBHEADER_LONG *) mac_pdu_ptr)->F = 1;
      ((NR_MAC_SUBHEADER_LONG *) mac_pdu_ptr)->LCID = sdu_lcids[i];
      ((NR_MAC_SUBHEADER_LONG *) mac_pdu_ptr)->L1 = ((unsigned short) sdu_lengths[i] >> 8) & 0xff;
      ((NR_MAC_SUBHEADER_LONG *) mac_pdu_ptr)->L2 = (unsigned short) sdu_lengths[i] & 0xff;
      last_size = 3;
    }

    mac_pdu_ptr += last_size;
    // 3) cycle through SDUs, compute each relevant and place dlsch_buffer in
    memcpy((void *) mac_pdu_ptr, (void *) dlsch_buffer_ptr, sdu_lengths[i]);
    dlsch_buffer_ptr += sdu_lengths[i];
    mac_pdu_ptr += sdu_lengths[i];
    LOG_D(MAC, "Generate DLSCH header num sdu %d len header %d len sdu %d -> offset %ld\n", num_sdus, last_size, sdu_lengths[i], (unsigned char *)mac_pdu_ptr - mac_pdu);
  }

  // 4) Compute final offset for padding
  if (post_padding > 0) {
    ((NR_MAC_SUBHEADER_FIXED *) mac_pdu_ptr)->R = 0;
    ((NR_MAC_SUBHEADER_FIXED *) mac_pdu_ptr)->LCID = DL_SCH_LCID_PADDING;
    mac_pdu_ptr++;
    LOG_D(MAC, "Generate Padding -> offset %ld\n", (unsigned char *)mac_pdu_ptr - mac_pdu);
  } else {
    // no MAC subPDU with padding
  }

  // compute final offset
  offset = ((unsigned char *) mac_pdu_ptr - mac_pdu);
  //printf("Offset %d \n", ((unsigned char *) mac_pdu_ptr - mac_pdu));
  return offset;
}

int getNrOfSymbols(NR_BWP_Downlink_t *bwp, int tda) {
  struct NR_PDSCH_TimeDomainResourceAllocationList *tdaList =
    bwp->bwp_Common->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList;
  AssertFatal(tda < tdaList->list.count,
              "time_domain_allocation %d>=%d\n",
              tda,
              tdaList->list.count);

  const int startSymbolAndLength = tdaList->list.array[tda]->startSymbolAndLength;
  int startSymbolIndex, nrOfSymbols;
  SLIV2SL(startSymbolAndLength, &startSymbolIndex, &nrOfSymbols);
  return nrOfSymbols;
}

nfapi_nr_dmrs_type_e getDmrsConfigType(NR_BWP_Downlink_t *bwp) {
  return bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->dmrs_Type == NULL ? 0 : 1;
}

uint8_t getN_PRB_DMRS(NR_BWP_Downlink_t *bwp, int numDmrsCdmGrpsNoData) {
  const nfapi_nr_dmrs_type_e dmrsConfigType = getDmrsConfigType(bwp);
  if (dmrsConfigType == NFAPI_NR_DMRS_TYPE1) {
    // if no data in dmrs cdm group is 1 only even REs have no data
    // if no data in dmrs cdm group is 2 both odd and even REs have no data
    return numDmrsCdmGrpsNoData * 6;
  } else {
    return numDmrsCdmGrpsNoData * 4;
  }
}

void nr_simple_dlsch_preprocessor(module_id_t module_id,
                                  frame_t frame,
                                  sub_frame_t slot) {
  NR_UE_info_t *UE_info = &RC.nrmac[module_id]->UE_info;

  AssertFatal(UE_info->num_UEs <= 1,
              "%s() cannot handle more than one UE, but found %d\n",
              __func__,
              UE_info->num_UEs);
  if (UE_info->num_UEs == 0)
    return;

  const int UE_id = 0;
  const int CC_id = 0;

  NR_UE_sched_ctrl_t *sched_ctrl = &UE_info->UE_sched_ctrl[UE_id];

  /* Retrieve amount of data to send for this UE */
  sched_ctrl->num_total_bytes = 0;
  const int lcid = DL_SCH_LCID_DTCH;
  const rnti_t rnti = UE_info->rnti[UE_id];
  sched_ctrl->rlc_status[lcid] = mac_rlc_status_ind(module_id,
                                                    rnti,
                                                    module_id,
                                                    frame,
                                                    slot,
                                                    ENB_FLAG_YES,
                                                    MBMS_FLAG_NO,
                                                    lcid,
                                                    0,
                                                    0);
  sched_ctrl->num_total_bytes += sched_ctrl->rlc_status[lcid].bytes_in_buffer;
  if (sched_ctrl->num_total_bytes == 0
      && !sched_ctrl->ta_apply) /* If TA should be applied, give at least one RB */
    return;

  LOG_D(MAC, "[%s][%d.%d], DTCH%d->DLSCH, RLC status %d bytes TA %d\n",
        __FUNCTION__,
        frame,
        slot,
        lcid,
        sched_ctrl->rlc_status[lcid].bytes_in_buffer,
        sched_ctrl->ta_apply);

  /* Find a free CCE */
  const int target_ss = NR_SearchSpace__searchSpaceType_PR_ue_Specific;
  sched_ctrl->search_space = get_searchspace(sched_ctrl->active_bwp, target_ss);
  uint8_t nr_of_candidates;
  find_aggregation_candidates(&sched_ctrl->aggregation_level,
                              &nr_of_candidates,
                              sched_ctrl->search_space);
  sched_ctrl->coreset = get_coreset(
      sched_ctrl->active_bwp, sched_ctrl->search_space, 1 /* dedicated */);
  int cid = sched_ctrl->coreset->controlResourceSetId;
  const uint16_t Y = UE_info->Y[UE_id][cid][slot];
  const int m = UE_info->num_pdcch_cand[UE_id][cid];
  sched_ctrl->cce_index = allocate_nr_CCEs(RC.nrmac[module_id],
                                           sched_ctrl->active_bwp,
                                           sched_ctrl->coreset,
                                           sched_ctrl->aggregation_level,
                                           Y,
                                           m,
                                           nr_of_candidates);
  if (sched_ctrl->cce_index < 0) {
    LOG_E(MAC, "%s(): could not find CCE for UE %d\n", __func__, UE_id);
    return;
  }
  UE_info->num_pdcch_cand[UE_id][cid]++;

  /* Find PUCCH occasion: if it fails, undo CCE allocation (undoing PUCCH
   * allocation after CCE alloc fail would be more complex) */
  const bool alloc = nr_acknack_scheduling(module_id, UE_id, frame, slot);
  if (!alloc) {
    LOG_W(MAC,
          "%s(): could not find PUCCH for UE %d/%04x@%d.%d\n",
          __func__,
          UE_id,
          rnti,
          frame,
          slot);
    UE_info->num_pdcch_cand[UE_id][cid]--;
    int *cce_list = RC.nrmac[module_id]->cce_list[sched_ctrl->active_bwp->bwp_Id][cid];
    for (int i = 0; i < sched_ctrl->aggregation_level; i++)
      cce_list[sched_ctrl->cce_index + i] = 0;
    return;
  }

  uint16_t *vrb_map = RC.nrmac[module_id]->common_channels[CC_id].vrb_map;
  // for now HARQ PID is fixed and should be the same as in post-processor
  const int current_harq_pid = slot % 8;
  NR_UE_harq_t *harq = &sched_ctrl->harq_processes[current_harq_pid];
  NR_UE_ret_info_t *retInfo = &sched_ctrl->retInfo[current_harq_pid];
  const uint16_t bwpSize = NRRIV2BW(sched_ctrl->active_bwp->bwp_Common->genericParameters.locationAndBandwidth, 275);
  int rbStart = NRRIV2PRBOFFSET(sched_ctrl->active_bwp->bwp_Common->genericParameters.locationAndBandwidth, 275);

  if (harq->round != 0) { /* retransmission */
    sched_ctrl->time_domain_allocation = retInfo->time_domain_allocation;

    /* ensure that there is a free place for RB allocation */
    int rbSize = 0;
    while (rbSize < retInfo->rbSize) {
      rbStart += rbSize; /* last iteration rbSize was not enough, skip it */
      rbSize = 0;
      while (rbStart < bwpSize && vrb_map[rbStart]) rbStart++;
      if (rbStart >= bwpSize) {
        LOG_E(MAC,
              "cannot allocate retransmission for UE %d/RNTI %04x: no resources\n",
              UE_id,
              rnti);
        return;
      }
      while (rbStart + rbSize < bwpSize
             && !vrb_map[rbStart + rbSize]
             && rbSize < retInfo->rbSize)
        rbSize++;
    }
    sched_ctrl->rbSize = retInfo->rbSize;
    sched_ctrl->rbStart = rbStart;

    /* MCS etc: just reuse from previous scheduling opportunity */
    sched_ctrl->mcsTableIdx = retInfo->mcsTableIdx;
    sched_ctrl->mcs = retInfo->mcs;
    sched_ctrl->numDmrsCdmGrpsNoData = retInfo->numDmrsCdmGrpsNoData;
  } else {
    // Time-domain allocation
    sched_ctrl->time_domain_allocation = 2;

    // modulation scheme
    sched_ctrl->mcsTableIdx = 0;
    sched_ctrl->mcs = 9;
    sched_ctrl->numDmrsCdmGrpsNoData = 1;

    // Freq-demain allocation
    while (rbStart < bwpSize && vrb_map[rbStart]) rbStart++;

    uint8_t N_PRB_DMRS =
        getN_PRB_DMRS(sched_ctrl->active_bwp, sched_ctrl->numDmrsCdmGrpsNoData);
    int nrOfSymbols = getNrOfSymbols(sched_ctrl->active_bwp,
                                     sched_ctrl->time_domain_allocation);
    uint8_t N_DMRS_SLOT = get_num_dmrs_symbols(sched_ctrl->active_bwp->bwp_Dedicated->pdsch_Config->choice.setup,
                                               RC.nrmac[module_id]->common_channels->ServingCellConfigCommon->dmrs_TypeA_Position ,
                                               nrOfSymbols);

    int rbSize = 0;
    const int oh = 2 + (sched_ctrl->num_total_bytes >= 256)
                 + 2 * (frame == (sched_ctrl->ta_frame + 10) % 1024);
    uint32_t TBS = 0;
    do {
      rbSize++;
      TBS = nr_compute_tbs(nr_get_Qm_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx),
                           nr_get_code_rate_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx),
                           rbSize,
                           nrOfSymbols,
                           N_PRB_DMRS * N_DMRS_SLOT,
                           0 /* N_PRB_oh, 0 for initialBWP */,
                           0 /* tb_scaling */,
                           1 /* nrOfLayers */)
            >> 3;
    } while (rbStart + rbSize < bwpSize && !vrb_map[rbStart + rbSize] && TBS < sched_ctrl->num_total_bytes + oh);
    sched_ctrl->rbSize = rbSize;
    sched_ctrl->rbStart = rbStart;
  }

  /* mark the corresponding RBs as used */
  for (int rb = 0; rb < sched_ctrl->rbSize; rb++)
    vrb_map[rb + sched_ctrl->rbStart] = 1;
}

void nr_schedule_ue_spec(module_id_t module_id,
                         frame_t frame,
                         sub_frame_t slot) {
  gNB_MAC_INST *gNB_mac = RC.nrmac[module_id];

  /* PREPROCESSOR */
  gNB_mac->pre_processor_dl(module_id, frame, slot);

  const int CC_id = 0;
  NR_ServingCellConfigCommon_t *scc = gNB_mac->common_channels[CC_id].ServingCellConfigCommon;
  NR_UE_info_t *UE_info = &gNB_mac->UE_info;

  NR_UE_list_t *UE_list = &UE_info->list;
  for (int UE_id = UE_list->head; UE_id >= 0; UE_id = UE_list->next[UE_id]) {
    NR_UE_sched_ctrl_t *sched_ctrl = &UE_info->UE_sched_ctrl[UE_id];

    /* update TA and set ta_apply every 10 frames.
     * Possible improvement: take the periodicity from input file.
     * If such UE is not scheduled now, it will be by the preprocessor later.
     * If we add the CE, ta_apply will be reset */
    if (frame == (sched_ctrl->ta_frame + 10) % 1024){
      sched_ctrl->ta_apply = true; /* the timer is reset once TA CE is scheduled */
      LOG_D(MAC, "[UE %d][%d.%d] UL timing alignment procedures: setting flag for Timing Advance command\n", UE_id, frame, slot);
    }

    if (sched_ctrl->rbSize <= 0)
      continue;

    const rnti_t rnti = UE_info->rnti[UE_id];

    /* POST processing */
    struct NR_PDSCH_TimeDomainResourceAllocationList *tdaList =
      sched_ctrl->active_bwp->bwp_Common->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList;
    AssertFatal(sched_ctrl->time_domain_allocation < tdaList->list.count,
                "time_domain_allocation %d>=%d\n",
                sched_ctrl->time_domain_allocation,
                tdaList->list.count);

    const int startSymbolAndLength =
      tdaList->list.array[sched_ctrl->time_domain_allocation]->startSymbolAndLength;
    int startSymbolIndex, nrOfSymbols;
    SLIV2SL(startSymbolAndLength, &startSymbolIndex, &nrOfSymbols);

    uint8_t N_PRB_DMRS =
        getN_PRB_DMRS(sched_ctrl->active_bwp, sched_ctrl->numDmrsCdmGrpsNoData);
    uint8_t N_DMRS_SLOT = get_num_dmrs_symbols(sched_ctrl->active_bwp->bwp_Dedicated->pdsch_Config->choice.setup,
                                               RC.nrmac[module_id]->common_channels->ServingCellConfigCommon->dmrs_TypeA_Position ,
                                               nrOfSymbols);
    const nfapi_nr_dmrs_type_e dmrsConfigType = getDmrsConfigType(sched_ctrl->active_bwp);
    const int nrOfLayers = 1;
    const uint16_t R = nr_get_code_rate_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx);
    const uint8_t Qm = nr_get_Qm_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx);
    const uint32_t TBS =
        nr_compute_tbs(nr_get_Qm_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx),
                       nr_get_code_rate_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx),
                       sched_ctrl->rbSize,
                       nrOfSymbols,
                       N_PRB_DMRS * N_DMRS_SLOT,
                       0 /* N_PRB_oh, 0 for initialBWP */,
                       0 /* tb_scaling */,
                       nrOfLayers)
        >> 3;

    const int current_harq_pid = slot % 8;
    NR_UE_harq_t *harq = &sched_ctrl->harq_processes[current_harq_pid];
    NR_sched_pucch_t *pucch = &sched_ctrl->sched_pucch[0];
    harq->feedback_slot = pucch->ul_slot;
    harq->is_waiting = 1;
    UE_info->mac_stats[UE_id].dlsch_rounds[harq->round]++;

    LOG_D(MAC,
          "%4d.%2d RNTI %04x start %d RBs %d startSymbol %d nb_symbsol %d MCS %d TBS %d HARQ PID %d round %d NDI %d\n",
          frame,
          slot,
          rnti,
          sched_ctrl->rbStart,
          sched_ctrl->rbSize,
          startSymbolIndex,
          nrOfSymbols,
          sched_ctrl->mcs,
          TBS,
          current_harq_pid,
          harq->round,
          harq->ndi);

    NR_BWP_Downlink_t *bwp = sched_ctrl->active_bwp;
    AssertFatal(bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList,
                "searchSpacesToAddModList is null\n");
    AssertFatal(bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList->list.count > 0,
                "searchSPacesToAddModList is empty\n");


    nfapi_nr_dl_tti_request_body_t *dl_req = &gNB_mac->DL_req[CC_id].dl_tti_request_body;

    /* TODO: can be moved down? */
    nfapi_nr_dl_tti_request_pdu_t *dl_tti_pdcch_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs];
    memset(dl_tti_pdcch_pdu, 0, sizeof(nfapi_nr_dl_tti_request_pdu_t));
    dl_tti_pdcch_pdu->PDUType = NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE;
    dl_tti_pdcch_pdu->PDUSize = (uint8_t)(2+sizeof(nfapi_nr_dl_tti_pdcch_pdu));
    dl_req->nPDUs += 1;
    nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu = &dl_tti_pdcch_pdu->pdcch_pdu.pdcch_pdu_rel15;

    nfapi_nr_dl_tti_request_pdu_t *dl_tti_pdsch_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs];
    memset(dl_tti_pdsch_pdu, 0, sizeof(nfapi_nr_dl_tti_request_pdu_t));
    dl_tti_pdsch_pdu->PDUType = NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE;
    dl_tti_pdsch_pdu->PDUSize = (uint8_t)(2+sizeof(nfapi_nr_dl_tti_pdsch_pdu));
    dl_req->nPDUs += 1;
    nfapi_nr_dl_tti_pdsch_pdu_rel15_t *pdsch_pdu = &dl_tti_pdsch_pdu->pdsch_pdu.pdsch_pdu_rel15;

    pdsch_pdu->pduBitmap = 0;
    pdsch_pdu->rnti = rnti;
    pdsch_pdu->pduIndex = gNB_mac->pdu_index[CC_id]++;

    // BWP
    pdsch_pdu->BWPSize  = NRRIV2BW(bwp->bwp_Common->genericParameters.locationAndBandwidth, 275);
    pdsch_pdu->BWPStart = NRRIV2PRBOFFSET(bwp->bwp_Common->genericParameters.locationAndBandwidth,275);
    pdsch_pdu->SubcarrierSpacing = bwp->bwp_Common->genericParameters.subcarrierSpacing;
    if (bwp->bwp_Common->genericParameters.cyclicPrefix)
      pdsch_pdu->CyclicPrefix = *bwp->bwp_Common->genericParameters.cyclicPrefix;
    else
      pdsch_pdu->CyclicPrefix = 0;

    // Codeword information
    pdsch_pdu->NrOfCodewords = 1;
    pdsch_pdu->targetCodeRate[0] = R;
    pdsch_pdu->qamModOrder[0] = Qm;
    pdsch_pdu->mcsIndex[0] = sched_ctrl->mcs;
    pdsch_pdu->mcsTable[0] = sched_ctrl->mcsTableIdx;
    pdsch_pdu->rvIndex[0] = nr_rv_round_map[harq->round];
    pdsch_pdu->TBSize[0] = TBS;

    pdsch_pdu->dataScramblingId = *scc->physCellId;
    pdsch_pdu->nrOfLayers = nrOfLayers;
    pdsch_pdu->transmissionScheme = 0;
    pdsch_pdu->refPoint = 0; // Point A

    // DMRS
    pdsch_pdu->dlDmrsSymbPos =
        fill_dmrs_mask(bwp->bwp_Dedicated->pdsch_Config->choice.setup,
                       scc->dmrs_TypeA_Position,
                       nrOfSymbols);
    pdsch_pdu->dmrsConfigType = dmrsConfigType;
    pdsch_pdu->dlDmrsScramblingId = *scc->physCellId;
    pdsch_pdu->SCID = 0;
    pdsch_pdu->numDmrsCdmGrpsNoData = sched_ctrl->numDmrsCdmGrpsNoData;
    pdsch_pdu->dmrsPorts = 1;

    // Pdsch Allocation in frequency domain
    pdsch_pdu->resourceAlloc = 1;
    pdsch_pdu->rbStart = sched_ctrl->rbStart;
    pdsch_pdu->rbSize = sched_ctrl->rbSize;
    pdsch_pdu->VRBtoPRBMapping = 1; // non-interleaved, check if this is ok for initialBWP

    // Resource Allocation in time domain
    pdsch_pdu->StartSymbolIndex = startSymbolIndex;
    pdsch_pdu->NrOfSymbols = nrOfSymbols;

    /* Check and validate PTRS values */
    struct NR_SetupRelease_PTRS_DownlinkConfig *phaseTrackingRS =
        bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->phaseTrackingRS;
    if (phaseTrackingRS) {
      bool valid_ptrs_setup = set_dl_ptrs_values(phaseTrackingRS->choice.setup,
                                                 pdsch_pdu->rbSize,
                                                 pdsch_pdu->mcsIndex[0],
                                                 pdsch_pdu->mcsTable[0],
                                                 &pdsch_pdu->PTRSFreqDensity,
                                                 &pdsch_pdu->PTRSTimeDensity,
                                                 &pdsch_pdu->PTRSPortIndex,
                                                 &pdsch_pdu->nEpreRatioOfPDSCHToPTRS,
                                                 &pdsch_pdu->PTRSReOffset,
                                                 pdsch_pdu->NrOfSymbols);
      if (valid_ptrs_setup)
        pdsch_pdu->pduBitmap |= 0x1; // Bit 0: pdschPtrs - Indicates PTRS included (FR2)
    }

    dci_pdu_rel15_t dci_pdu[MAX_DCI_CORESET];
    memset(dci_pdu, 0, sizeof(dci_pdu_rel15_t) * MAX_DCI_CORESET);

    // bwp indicator
    const int n_dl_bwp = UE_info->secondaryCellGroup[UE_id]->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count;
      AssertFatal(n_dl_bwp == 1,
          "downlinkBWP_ToAddModList has %d BWP!\n",
          n_dl_bwp);
    // as per table 7.3.1.1.2-1 in 38.212
    dci_pdu[0].bwp_indicator.val = n_dl_bwp < 4 ? bwp->bwp_Id : bwp->bwp_Id - 1;
    AssertFatal(bwp->bwp_Dedicated->pdsch_Config->choice.setup->resourceAllocation == NR_PDSCH_Config__resourceAllocation_resourceAllocationType1,
                "Only frequency resource allocation type 1 is currently supported\n");
    dci_pdu[0].frequency_domain_assignment.val =
        PRBalloc_to_locationandbandwidth0(
            pdsch_pdu->rbSize,
            pdsch_pdu->rbStart,
            pdsch_pdu->BWPSize);
    dci_pdu[0].time_domain_assignment.val = sched_ctrl->time_domain_allocation;
    dci_pdu[0].mcs = sched_ctrl->mcs;
    dci_pdu[0].rv = pdsch_pdu->rvIndex[0];
    dci_pdu[0].harq_pid = current_harq_pid;
    dci_pdu[0].ndi = harq->ndi;
    dci_pdu[0].dai[0].val = (pucch->dai_c-1)&3;
    dci_pdu[0].tpc = sched_ctrl->tpc1; // TPC for PUCCH: table 7.2.1-1 in 38.213
    dci_pdu[0].pucch_resource_indicator = pucch->resource_indicator;
    dci_pdu[0].pdsch_to_harq_feedback_timing_indicator.val = pucch->timing_indicator; // PDSCH to HARQ TI
    dci_pdu[0].antenna_ports.val = 0;  // nb of cdm groups w/o data 1 and dmrs port 0
    dci_pdu[0].dmrs_sequence_initialization.val = pdsch_pdu->SCID;
    LOG_D(MAC,
          "%4d.%2d DCI type 1 payload: freq_alloc %d (%d,%d,%d), "
          "time_alloc %d, vrb to prb %d, mcs %d tb_scaling %d ndi %d rv %d\n",
          frame,
          slot,
          dci_pdu[0].frequency_domain_assignment.val,
          pdsch_pdu->rbStart,
          pdsch_pdu->rbSize,
          pdsch_pdu->BWPSize,
          dci_pdu[0].time_domain_assignment.val,
          dci_pdu[0].vrb_to_prb_mapping.val,
          dci_pdu[0].mcs,
          dci_pdu[0].tb_scaling,
          dci_pdu[0].ndi,
          dci_pdu[0].rv);

    nr_configure_pdcch(gNB_mac,
                       pdcch_pdu,
                       rnti,
                       sched_ctrl->search_space,
                       sched_ctrl->coreset,
                       scc,
                       bwp,
                       sched_ctrl->aggregation_level,
                       sched_ctrl->cce_index);

    const long f = sched_ctrl->search_space->searchSpaceType->choice.ue_Specific->dci_Formats;
    const int dci_format = f ? NR_DL_DCI_FORMAT_1_1 : NR_DL_DCI_FORMAT_1_0;
    const int rnti_type = NR_RNTI_C;

    // nr_configure_pdcch() increased numDlDci, so we use numDlDci - 1
    fill_dci_pdu_rel15(scc,
                       UE_info->secondaryCellGroup[UE_id],
                       &pdcch_pdu->dci_pdu[pdcch_pdu->numDlDci - 1],
                       dci_pdu,
                       dci_format,
                       rnti_type,
                       pdsch_pdu->BWPSize,
                       bwp->bwp_Id);

    LOG_D(MAC,
          "coreset params: FreqDomainResource %llx, start_symbol %d  n_symb %d\n",
          (unsigned long long)pdcch_pdu->FreqDomainResource,
          pdcch_pdu->StartSymbolIndex,
          pdcch_pdu->DurationSymbols);

    NR_UE_ret_info_t *retInfo = &sched_ctrl->retInfo[current_harq_pid];
    if (harq->round != 0) { /* retransmission */
      if (sched_ctrl->rbSize != retInfo->rbSize)
        LOG_W(MAC,
              "retransmission uses different rbSize (%d vs. orig %d)\n",
              sched_ctrl->rbSize,
              retInfo->rbSize);
      if (sched_ctrl->time_domain_allocation != retInfo->time_domain_allocation)
        LOG_W(MAC,
              "retransmission uses different time_domain_allocation (%d vs. orig %d)\n",
              sched_ctrl->time_domain_allocation,
              retInfo->time_domain_allocation);
      if (sched_ctrl->mcs != retInfo->mcs
          || sched_ctrl->mcsTableIdx != retInfo->mcsTableIdx
          || sched_ctrl->numDmrsCdmGrpsNoData != retInfo->numDmrsCdmGrpsNoData)
        LOG_W(MAC,
              "retransmission uses different table/MCS/numDmrsCdmGrpsNoData (%d/%d/%d vs. orig %d/%d/%d)\n",
              sched_ctrl->mcsTableIdx,
              sched_ctrl->mcs,
              sched_ctrl->numDmrsCdmGrpsNoData,
              retInfo->mcsTableIdx,
              retInfo->mcs,
              retInfo->numDmrsCdmGrpsNoData);
      /* we do not have to do anything, since we do not require to get data
       * from RLC, encode MAC CEs, or copy data to FAPI structures */
      LOG_W(MAC,
            "%d.%2d DL retransmission UE %d/RNTI %04x HARQ PID %d round %d NDI %d\n",
            frame,
            slot,
            UE_id,
            rnti,
            current_harq_pid,
            harq->round,
            harq->ndi);
    } else { /* initial transmission */

      LOG_D(MAC, "[%s] Initial HARQ transmission in %d.%d\n", __FUNCTION__, frame, slot);
      /* reserve space for timing advance of UE if necessary,
       * nr_generate_dlsch_pdu() checks for ta_apply and add TA CE if necessary */
      const int ta_len = (sched_ctrl->ta_apply) ? 2 : 0;

      /* Get RLC data */
      int header_length_total = 0;
      int header_length_last = 0;
      int sdu_length_total = 0;
      int num_sdus = 0;
      uint16_t sdu_lengths[NB_RB_MAX] = {0};
      uint8_t mac_sdus[MAX_NR_DLSCH_PAYLOAD_BYTES];
      unsigned char sdu_lcids[NB_RB_MAX] = {0};
      const int lcid = DL_SCH_LCID_DTCH;
      if (sched_ctrl->num_total_bytes > 0) {
        /* this is the data from the RLC we would like to request (e.g., only
         * some bytes for first LC and some more from a second one */
        const rlc_buffer_occupancy_t ndata = sched_ctrl->rlc_status[lcid].bytes_in_buffer;
        /* this is the maximum data we can transport based on TBS minus headers */
        const int mindata = min(ndata, TBS - ta_len - header_length_total - sdu_length_total -  2 - (ndata >= 256));
        LOG_D(MAC,
              "[gNB %d][USER-PLANE DEFAULT DRB] Frame %d : DTCH->DLSCH, Requesting "
              "%d bytes from RLC (lcid %d total hdr len %d), TBS: %d \n \n",
              module_id,
              frame,
              mindata,
              lcid,
              header_length_total,
              TBS);

        sdu_lengths[num_sdus] = mac_rlc_data_req(module_id,
            rnti,
            module_id,
            frame,
            ENB_FLAG_YES,
            MBMS_FLAG_NO,
            lcid,
            mindata,
            (char *)&mac_sdus[sdu_length_total],
            0,
            0);

        LOG_D(MAC,
              "[gNB %d][USER-PLANE DEFAULT DRB] Got %d bytes for DTCH %d \n",
              module_id,
              sdu_lengths[num_sdus],
              lcid);

        sdu_lcids[num_sdus] = lcid;
        sdu_length_total += sdu_lengths[num_sdus];
        header_length_last = 1 + 1 + (sdu_lengths[num_sdus] >= 128);
        header_length_total += header_length_last;
        num_sdus++;
      }
      else if (get_softmodem_params()->phy_test || get_softmodem_params()->do_ra) {
        LOG_D(MAC, "Configuring DL_TX in %d.%d: random data\n", frame, slot);
        // fill dlsch_buffer with random data
        for (int i = 0; i < TBS; i++)
          mac_sdus[i] = (unsigned char) (lrand48()&0xff);
        sdu_lcids[0] = 0x3f; // DRB
        sdu_lengths[0] = TBS - ta_len - 3;
        header_length_total += 2 + (sdu_lengths[0] >= 256);
        sdu_length_total += sdu_lengths[0];
        num_sdus +=1;
      }

      UE_info->mac_stats[UE_id].dlsch_total_bytes += TBS;
      UE_info->mac_stats[UE_id].lc_bytes_tx[lcid] += sdu_length_total;

      const int post_padding = TBS > header_length_total + sdu_length_total + ta_len;

      const int ntx_req = gNB_mac->TX_req[CC_id].Number_of_PDUs;
      nfapi_nr_pdu_t *tx_req = &gNB_mac->TX_req[CC_id].pdu_list[ntx_req];
      /* pointer to directly generate the PDU into the nFAPI structure */
      uint32_t *buf = tx_req->TLVs[0].value.direct;

      const int offset = nr_generate_dlsch_pdu(
          module_id,
          sched_ctrl,
          (unsigned char *)mac_sdus,
          (unsigned char *)buf,
          num_sdus, // num_sdus
          sdu_lengths,
          sdu_lcids,
          255, // no drx
          NULL, // contention res id
          post_padding);

      // Padding: fill remainder of DLSCH with 0
      if (post_padding > 0) {
        for (int j = 0; j < TBS - offset; j++)
          buf[offset + j] = 0;
      }

      /* the buffer has been filled by nr_generate_dlsch_pdu(), below we simply
       * fill the remaining information */
      tx_req->PDU_length = TBS;
      tx_req->PDU_index  = gNB_mac->pdu_index[0]++;
      tx_req->num_TLV = 1;
      tx_req->TLVs[0].length = TBS + 2;
      gNB_mac->TX_req[CC_id].Number_of_PDUs++;
      gNB_mac->TX_req[CC_id].SFN = frame;
      gNB_mac->TX_req[CC_id].Slot = slot;

      retInfo->rbSize = sched_ctrl->rbSize;
      retInfo->time_domain_allocation = sched_ctrl->time_domain_allocation;
      retInfo->mcsTableIdx = sched_ctrl->mcsTableIdx;
      retInfo->mcs = sched_ctrl->mcs;
      retInfo->numDmrsCdmGrpsNoData = sched_ctrl->numDmrsCdmGrpsNoData;

      // ta command is sent, values are reset
      if (sched_ctrl->ta_apply) {
        sched_ctrl->ta_apply = false;
        sched_ctrl->ta_frame = frame;
        LOG_D(MAC,
              "%d.%2d UE %d TA scheduled, resetting TA frame\n",
              frame,
              slot,
              UE_id);
      }

      T(T_GNB_MAC_DL_PDU_WITH_DATA, T_INT(module_id), T_INT(CC_id), T_INT(rnti),
        T_INT(frame), T_INT(slot), T_INT(current_harq_pid), T_BUFFER(buf, TBS));

#if defined(ENABLE_MAC_PAYLOAD_DEBUG)
      if (frame%100 == 0) {
        LOG_I(MAC,
              "%d.%d, first 10 payload bytes, TBS size: %d \n",
              frame,
              slot,
              TBS);
        for(int i = 0; i < 10; i++)
          LOG_I(MAC, "byte %d: %x\n", i, ((uint8_t *) buf)[i]);
      }
#endif
    }

    /* mark UE as scheduled */
    sched_ctrl->rbSize = 0;
  }
}
