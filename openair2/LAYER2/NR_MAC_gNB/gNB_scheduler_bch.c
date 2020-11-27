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

/*! \file gNB_scheduler_bch.c
 * \brief procedures related to eNB for the BCH transport channel
 * \author  Navid Nikaein and Raymond Knopp, WEI-TAI CHEN
 * \date 2010 - 2014, 2018
 * \email: navid.nikaein@eurecom.fr, kroempa@gmail.com
 * \version 1.0
 * \company Eurecom, NTUST
 * @ingroup _mac

 */

#include "assertions.h"
#include "NR_MAC_gNB/nr_mac_gNB.h"
#include "NR_MAC_gNB/mac_proto.h"
#include "NR_MAC_COMMON/nr_mac_extern.h"
#include "common/utils/LOG/log.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "UTIL/OPT/opt.h"
#include "OCG.h"
#include "OCG_extern.h"

#include "RRC/NR/nr_rrc_extern.h"


//#include "LAYER2/MAC/pre_processor.c"
#include "pdcp.h"

#include "intertask_interface.h"

#define ENABLE_MAC_PAYLOAD_DEBUG
#define DEBUG_eNB_SCHEDULER 1

#include "common/ran_context.h"

extern RAN_CONTEXT_t RC;
extern uint8_t SSB_Table[38];

void schedule_nr_mib(module_id_t module_idP, frame_t frameP, sub_frame_t slotP, uint8_t slots_per_frame){

  gNB_MAC_INST *gNB = RC.nrmac[module_idP];
  NR_COMMON_channels_t *cc;
  
  nfapi_nr_dl_tti_request_t      *dl_tti_request;
  nfapi_nr_dl_tti_request_body_t *dl_req;
  nfapi_nr_dl_tti_request_pdu_t  *dl_config_pdu;

  int mib_sdu_length;
  int CC_id;

  for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
    cc = &gNB->common_channels[CC_id];
    const long band = *cc->ServingCellConfigCommon->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.array[0];
    const uint32_t ssb_offset0 = *cc->ServingCellConfigCommon->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB - cc->ServingCellConfigCommon->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA;
    int ratio;
    switch (*cc->ServingCellConfigCommon->ssbSubcarrierSpacing) {
    case NR_SubcarrierSpacing_kHz15:
      AssertFatal(band <= 79,
                  "Band %ld is not possible for SSB with 15 kHz SCS\n",
                  band);
      if (band < 77)  // below 3GHz
        ratio = 3;    // NRARFCN step is 5 kHz
      else
        ratio = 1;  // NRARFCN step is 15 kHz
      break;
    case NR_SubcarrierSpacing_kHz30:
      AssertFatal(band <= 79,
                  "Band %ld is not possible for SSB with 15 kHz SCS\n",
                  band);
      if (band < 77)  // below 3GHz
        ratio = 6;    // NRARFCN step is 5 kHz
      else
        ratio = 2;  // NRARFCN step is 15 kHz
      break;
    case NR_SubcarrierSpacing_kHz120:
      AssertFatal(band >= 257,
                  "Band %ld is not possible for SSB with 120 kHz SCS\n",
                  band);
      ratio = 2;  // NRARFCN step is 15 kHz
      break;
    case NR_SubcarrierSpacing_kHz240:
      AssertFatal(band >= 257,
                  "Band %ld is not possible for SSB with 240 kHz SCS\n",
                  band);
      ratio = 4;  // NRARFCN step is 15 kHz
      break;
    default:
      AssertFatal(1 == 0, "SCS %ld not allowed for SSB \n",
                  *cc->ServingCellConfigCommon->ssbSubcarrierSpacing);
    }

    // scheduling MIB every 8 frames, PHY repeats it in between
    if((slotP == 0) && (frameP & 7) == 0) {
      dl_tti_request = &gNB->DL_req[CC_id];
      dl_req = &dl_tti_request->dl_tti_request_body;

      mib_sdu_length = mac_rrc_nr_data_req(module_idP, CC_id, frameP, MIBCH, 1, &cc->MIB_pdu.payload[0]);

      LOG_D(MAC, "Frame %d, slot %d: BCH PDU length %d\n", frameP, slotP, mib_sdu_length);

      if (mib_sdu_length > 0) {
        LOG_D(MAC,
              "Frame %d, slot %d: Adding BCH PDU in position %d (length %d)\n",
              frameP,
              slotP,
              dl_req->nPDUs,
              mib_sdu_length);

        if ((frameP & 1023) < 80){
          LOG_I(MAC,"[gNB %d] Frame %d : MIB->BCH  CC_id %d, Received %d bytes\n",module_idP, frameP, CC_id, mib_sdu_length);
        }

        dl_config_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs];
        memset((void *) dl_config_pdu, 0,sizeof(nfapi_nr_dl_tti_request_pdu_t));
        dl_config_pdu->PDUType      = NFAPI_NR_DL_TTI_SSB_PDU_TYPE;
        dl_config_pdu->PDUSize      =2 + sizeof(nfapi_nr_dl_tti_ssb_pdu_rel15_t);

        AssertFatal(cc->ServingCellConfigCommon->physCellId!=NULL,"cc->ServingCellConfigCommon->physCellId is null\n");
        dl_config_pdu->ssb_pdu.ssb_pdu_rel15.PhysCellId          = *cc->ServingCellConfigCommon->physCellId;
        dl_config_pdu->ssb_pdu.ssb_pdu_rel15.BetaPss             = 0;
        dl_config_pdu->ssb_pdu.ssb_pdu_rel15.SsbBlockIndex       = 0;
        AssertFatal(cc->ServingCellConfigCommon->downlinkConfigCommon!=NULL,"scc->downlinkConfigCommonL is null\n");
        AssertFatal(cc->ServingCellConfigCommon->downlinkConfigCommon->frequencyInfoDL!=NULL,"scc->downlinkConfigCommon->frequencyInfoDL is null\n");
        AssertFatal(cc->ServingCellConfigCommon->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB!=NULL,"scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB is null\n");
        AssertFatal(cc->ServingCellConfigCommon->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.count==1,"Frequency Band list does not have 1 element (%d)\n",cc->ServingCellConfigCommon->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.count);
        AssertFatal(cc->ServingCellConfigCommon->ssbSubcarrierSpacing,"ssbSubcarrierSpacing is null\n");
        AssertFatal(cc->ServingCellConfigCommon->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.array[0],"band is null\n");

        const nfapi_nr_config_request_scf_t *cfg = &RC.nrmac[module_idP]->config[0];
        dl_config_pdu->ssb_pdu.ssb_pdu_rel15.SsbSubcarrierOffset = cfg->ssb_table.ssb_subcarrier_offset.value; //kSSB
        dl_config_pdu->ssb_pdu.ssb_pdu_rel15.ssbOffsetPointA     = ssb_offset0/(ratio*12) - 10; // absoluteFrequencySSB is the center of SSB
        dl_config_pdu->ssb_pdu.ssb_pdu_rel15.bchPayloadFlag      = 1;
        dl_config_pdu->ssb_pdu.ssb_pdu_rel15.bchPayload          = (*(uint32_t*)cc->MIB_pdu.payload) & ((1<<24)-1);
        dl_req->nPDUs++;
      }
    }

    // checking if there is any SSB in slot
    const int abs_slot = (slots_per_frame * frameP) + slotP;
    const int slot_per_period = (slots_per_frame>>1)<<(*cc->ServingCellConfigCommon->ssb_periodicityServingCell);
    int eff_120_slot;
    const BIT_STRING_t *shortBitmap = &cc->ServingCellConfigCommon->ssb_PositionsInBurst->choice.shortBitmap;
    const BIT_STRING_t *mediumBitmap = &cc->ServingCellConfigCommon->ssb_PositionsInBurst->choice.mediumBitmap;
    const BIT_STRING_t *longBitmap = &cc->ServingCellConfigCommon->ssb_PositionsInBurst->choice.longBitmap;
    uint8_t buf = 0;
    switch (cc->ServingCellConfigCommon->ssb_PositionsInBurst->present) {
      case 1:
        // presence of ssbs possible in the first 2 slots of ssb period
        if ((abs_slot % slot_per_period) < 2 &&
            (((shortBitmap->buf[0]) >> (6 - (slotP << 1))) & 3) != 0)
          fill_ssb_vrb_map(cc, (ssb_offset0 / (ratio * 12) - 10), CC_id);
        break;
      case 2:
        // presence of ssbs possible in the first 4 slots of ssb period
        if ((abs_slot % slot_per_period) < 4 &&
            (((mediumBitmap->buf[0]) >> (6 - (slotP << 1))) & 3) != 0)
          fill_ssb_vrb_map(cc, (ssb_offset0 / (ratio * 12) - 10), CC_id);
        break;
      case 3:
        AssertFatal(*cc->ServingCellConfigCommon->ssbSubcarrierSpacing ==
                      NR_SubcarrierSpacing_kHz120,
                    "240kHZ subcarrier spacing currently not supported for SSBs\n");
        if ((abs_slot % slot_per_period) < 8) {
          eff_120_slot = slotP;
          buf = longBitmap->buf[0];
        } else if ((abs_slot % slot_per_period) < 17) {
          eff_120_slot = slotP - 9;
          buf = longBitmap->buf[1];
        } else if ((abs_slot % slot_per_period) < 26) {
          eff_120_slot = slotP - 18;
          buf = longBitmap->buf[2];
        } else if ((abs_slot % slot_per_period) < 35) {
          eff_120_slot = slotP - 27;
          buf = longBitmap->buf[3];
        }
        if (((buf >> (6 - (eff_120_slot << 1))) & 3) != 0)
          fill_ssb_vrb_map(cc, ssb_offset0 / (ratio * 12) - 10, CC_id);
        break;
    default:
      AssertFatal(0,
                  "SSB bitmap size value %d undefined (allowed values 1,2,3)\n",
                  cc->ServingCellConfigCommon->ssb_PositionsInBurst->present);
    }
  }
}


void schedule_nr_SI(module_id_t module_idP, frame_t frameP, sub_frame_t subframeP) {
//----------------------------------------  
}

void fill_ssb_vrb_map (NR_COMMON_channels_t *cc, int rbStart, int CC_id) {
  uint16_t *vrb_map = cc[CC_id].vrb_map;
  for (int rb = 0; rb < 20; rb++)
    vrb_map[rbStart + rb] = 1;
}
