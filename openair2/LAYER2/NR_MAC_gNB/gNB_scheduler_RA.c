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

/*! \file     gNB_scheduler_RA.c
 * \brief     primitives used for random access
 * \author    Guido Casati
 * \date      2019
 * \email:    guido.casati@iis.fraunhofer.de
 * \version
 */

#include "platform_types.h"

/* MAC */
#include "nr_mac_gNB.h"
#include "NR_MAC_gNB/mac_proto.h"
#include "NR_MAC_COMMON/nr_mac_extern.h"

/* Utils */
#include "common/utils/LOG/log.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/utils/nr/nr_common.h"
#include "UTIL/OPT/opt.h"
#include "SIMULATION/TOOLS/sim.h" // for taus

extern RAN_CONTEXT_t RC;
extern const uint8_t nr_slots_per_frame[5];

uint8_t DELTA[4]= {2,3,4,6};

#define MAX_NUMBER_OF_SSB 64		
float ssb_per_rach_occasion[8] = {0.125,0.25,0.5,1,2,4,8};

int16_t ssb_index_from_prach(module_id_t module_idP,
                             frame_t frameP,
			     sub_frame_t slotP,
			     uint16_t preamble_index,
			     uint8_t freq_index,
			     uint8_t symbol) {
  
  gNB_MAC_INST *gNB = RC.nrmac[module_idP];
  NR_COMMON_channels_t *cc = &gNB->common_channels[0];
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  nfapi_nr_config_request_scf_t *cfg = &RC.nrmac[module_idP]->config[0];

  uint8_t config_index = scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->rach_ConfigGeneric.prach_ConfigurationIndex;
	uint8_t fdm = cfg->prach_config.num_prach_fd_occasions.value;
  
	uint8_t total_RApreambles = 64;
	if( scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->totalNumberOfRA_Preambles != NULL)
    total_RApreambles = *scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->totalNumberOfRA_Preambles;	
  
	float  num_ssb_per_RO = ssb_per_rach_occasion[cfg->prach_config.ssb_per_rach.value];	
  uint16_t start_symbol_index = 0;
  uint8_t mu,N_dur,N_t_slot,start_symbol = 0, temp_start_symbol = 0, N_RA_slot;
  uint16_t format,RA_sfn_index = -1;
	uint8_t config_period = 1;
  uint16_t prach_occasion_id = -1;
	uint8_t num_active_ssb = cc->num_active_ssb;

  if (scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg1_SubcarrierSpacing)
    mu = *scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg1_SubcarrierSpacing;
  else
    mu = scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[0]->subcarrierSpacing;

  get_nr_prach_info_from_index(config_index,
			       (int)frameP,
			       (int)slotP,
			       scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA,
			       mu,
			       cc->frame_type,
			       &format,
			       &start_symbol,
			       &N_t_slot,
			       &N_dur,
			       &RA_sfn_index,
			       &N_RA_slot,
			       &config_period);
  uint8_t index = 0,slot_index = 0;
	for (slot_index = 0;slot_index < N_RA_slot; slot_index++) {
    if (N_RA_slot <= 1) { //1 PRACH slot in a subframe
       if((mu == 1) || (mu == 3))
         slot_index = 1;  //For scs = 30khz and 120khz
    }
    for (int i=0; i< N_t_slot; i++) {
      temp_start_symbol = (start_symbol + i * N_dur + 14 * slot_index) % 14;
		  if(symbol == temp_start_symbol) {
			  start_symbol_index = i;
		    break;
		  }
	  }
	}
  if (N_RA_slot <= 1) { //1 PRACH slot in a subframe
    if((mu == 1) || (mu == 3))
      slot_index = 0;  //For scs = 30khz and 120khz
  }
  
//  prach_occasion_id = subframe_index * N_t_slot * N_RA_slot * fdm + N_RA_slot_index * N_t_slot * fdm + freq_index + fdm * start_symbol_index; 
 prach_occasion_id = (((frameP % (cc->max_association_period * config_period))/config_period)*cc->total_prach_occasions_per_config_period) + (RA_sfn_index + slot_index) * N_t_slot * fdm + start_symbol_index * fdm + freq_index; 
//one RO is shared by one or more SSB
 if(num_ssb_per_RO <= 1 )
   index = (int) (prach_occasion_id / (int)(1/num_ssb_per_RO)) % num_active_ssb;
//one SSB have more than one continuous RO
 else if ( num_ssb_per_RO > 1) {
	 index = (prach_occasion_id * (int)num_ssb_per_RO)% num_active_ssb ;
   for(int j = 0;j < num_ssb_per_RO;j++) {
     if(preamble_index <  (((j+1) * total_RApreambles) / num_ssb_per_RO))
      index = index + j;
	  }		
	}

  LOG_D(MAC, "Frame %d, Slot %d: Prach Occasion id = %d ssb per RO = %f number of active SSB %u index = %d fdm %u symbol index %u freq_index %u total_RApreambles %u\n", frameP, slotP, prach_occasion_id, num_ssb_per_RO, num_active_ssb, index, fdm, start_symbol_index, freq_index, total_RApreambles);
  return index;
}
//Compute Total active SSBs and RO available
void find_SSB_and_RO_available(module_id_t module_idP) {

	gNB_MAC_INST *gNB = RC.nrmac[module_idP];
  NR_COMMON_channels_t *cc = &gNB->common_channels[0];
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  nfapi_nr_config_request_scf_t *cfg = &RC.nrmac[module_idP]->config[0];

  uint8_t config_index = scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->rach_ConfigGeneric.prach_ConfigurationIndex;
  uint8_t mu,N_dur,N_t_slot,start_symbol,N_RA_slot = 0;
  uint16_t format,N_RA_sfn = 0,unused_RA_occasion,repetition = 0;
	uint8_t num_active_ssb = 0;
  uint8_t max_association_period = 1;

  if (scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg1_SubcarrierSpacing)
    mu = *scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg1_SubcarrierSpacing;
  else
    mu = scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[0]->subcarrierSpacing;

  // prach is scheduled according to configuration index and tables 6.3.3.2.2 to 6.3.3.2.4
  get_nr_prach_occasion_info_from_index(config_index,
                                    scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA,
                                    mu,
                                    cc->frame_type,
                                    &format,
                                    &start_symbol,
                                    &N_t_slot,
                                    &N_dur,
                                    &N_RA_slot,
	                                 &N_RA_sfn,
                                   &max_association_period);

  float num_ssb_per_RO = ssb_per_rach_occasion[cfg->prach_config.ssb_per_rach.value];	
	uint8_t fdm = cfg->prach_config.num_prach_fd_occasions.value;
  uint64_t L_ssb = (((uint64_t) cfg->ssb_table.ssb_mask_list[0].ssb_mask.value)<<32) | cfg->ssb_table.ssb_mask_list[1].ssb_mask.value ;
	uint32_t total_RA_occasions = N_RA_sfn * N_t_slot * N_RA_slot * fdm;

	for(int i = 0;i < 64;i++) {
    if ((L_ssb >> (63-i)) & 0x01) { // only if the bit of L_ssb at current ssb index is 1
      cc->ssb_index[num_active_ssb] = i; 
		  num_active_ssb++;
    }
	}	

	for(int i = 1; (1 << (i-1)) <= max_association_period;i++) {
    if(total_RA_occasions >= (int) (num_active_ssb/num_ssb_per_RO)) {
		  repetition = (uint16_t)((total_RA_occasions * num_ssb_per_RO )/num_active_ssb);
		  break;
		} 
		else { 
		  total_RA_occasions = total_RA_occasions * i;
		  cc->max_association_period = i;
		} 
	}
  if(cc->max_association_period == 0)
			cc->max_association_period = 1;

 unused_RA_occasion = total_RA_occasions - (int)((num_active_ssb * repetition)/num_ssb_per_RO);
 cc->total_prach_occasions = total_RA_occasions - unused_RA_occasion;
 cc->num_active_ssb = num_active_ssb;

  LOG_I(MAC, "Total available RO %d, num of active SSB %d: unused RO = %d max_association_period %u N_RA_sfn %u \n", cc->total_prach_occasions, cc->num_active_ssb, unused_RA_occasion, max_association_period, N_RA_sfn);

}		
		
void schedule_nr_prach(module_id_t module_idP, frame_t frameP, sub_frame_t slotP) {

  gNB_MAC_INST *gNB = RC.nrmac[module_idP];
  NR_COMMON_channels_t *cc = gNB->common_channels;
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  nfapi_nr_ul_tti_request_t *UL_tti_req = &RC.nrmac[module_idP]->UL_tti_req[0];
  nfapi_nr_config_request_scf_t *cfg = &RC.nrmac[module_idP]->config[0];

  if (is_nr_UL_slot(scc,slotP)) {
  uint8_t config_index = scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->rach_ConfigGeneric.prach_ConfigurationIndex;
  uint8_t mu,N_dur,N_t_slot,start_symbol = 0,N_RA_slot;
  uint16_t RA_sfn_index = -1;
	uint8_t config_period = 1;
  uint16_t format;
  int slot_index = 0;
  uint16_t prach_occasion_id = -1;

  if (scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg1_SubcarrierSpacing)
    mu = *scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg1_SubcarrierSpacing;
  else
    mu = scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[0]->subcarrierSpacing;

  uint8_t fdm = cfg->prach_config.num_prach_fd_occasions.value;
  // prach is scheduled according to configuration index and tables 6.3.3.2.2 to 6.3.3.2.4
  if ( get_nr_prach_info_from_index(config_index,
                                    (int)frameP,
                                    (int)slotP,
                                    scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA,
                                    mu,
                                    cc->frame_type,
                                    &format,
                                    &start_symbol,
                                    &N_t_slot,
                                    &N_dur,
                                    &RA_sfn_index,
                                    &N_RA_slot,
																		&config_period) ) {
    uint16_t format0 = format&0xff;      // first column of format from table
    uint16_t format1 = (format>>8)&0xff; // second column of format from table

    if (N_RA_slot > 1) { //more than 1 PRACH slot in a subframe
      if (slotP%2 == 1){
	      slot_index = 1;
      }	
      else {
	      slot_index = 0;
			}	
    }else if (N_RA_slot <= 1) { //1 PRACH slot in a subframe
       slot_index = 0;
    }


    UL_tti_req->SFN = frameP;
    UL_tti_req->Slot = slotP;
    for (int fdm_index=0; fdm_index < fdm; fdm_index++) { // one structure per frequency domain occasion
    for (int td_index=0; td_index<N_t_slot; td_index++) {

      prach_occasion_id = (((frameP % (cc->max_association_period * config_period))/config_period) * cc->total_prach_occasions_per_config_period) + (RA_sfn_index + slot_index) * N_t_slot * fdm + td_index * fdm + fdm_index;
			if((prach_occasion_id < cc->total_prach_occasions) && (td_index == 0)){  

      UL_tti_req->pdus_list[UL_tti_req->n_pdus].pdu_type = NFAPI_NR_UL_CONFIG_PRACH_PDU_TYPE;
      UL_tti_req->pdus_list[UL_tti_req->n_pdus].pdu_size = sizeof(nfapi_nr_prach_pdu_t);
      nfapi_nr_prach_pdu_t  *prach_pdu = &UL_tti_req->pdus_list[UL_tti_req->n_pdus].prach_pdu;
      memset(prach_pdu,0,sizeof(nfapi_nr_prach_pdu_t));
      UL_tti_req->n_pdus+=1;

      // filling the prach fapi structure
      prach_pdu->phys_cell_id = *scc->physCellId;
      prach_pdu->num_prach_ocas = N_t_slot;
      prach_pdu->prach_start_symbol = start_symbol;
      prach_pdu->num_ra = fdm_index;
      prach_pdu->num_cs = get_NCS(scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->rach_ConfigGeneric.zeroCorrelationZoneConfig,
                                  format0,
                                  scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->restrictedSetConfig);
      
      LOG_D(MAC, "Frame %d, Slot %d: Prach Occasion id = %u  fdm index = %u start symbol = %u slot index = %u subframe index = %u \n",
	    frameP, slotP,
	    prach_occasion_id, prach_pdu->num_ra,
	    prach_pdu->prach_start_symbol,
	    slot_index, RA_sfn_index);
      // SCF PRACH PDU format field does not consider A1/B1 etc. possibilities
      // We added 9 = A1/B1 10 = A2/B2 11 A3/B3
      if (format1!=0xff) {
        switch(format0) {
          case 0xa1:
            prach_pdu->prach_format = 11;
            break;
          case 0xa2:
            prach_pdu->prach_format = 12;
            break;
          case 0xa3:
            prach_pdu->prach_format = 13;
            break;
        default:
          AssertFatal(1==0,"Only formats A1/B1 A2/B2 A3/B3 are valid for dual format");
        }
      }
      else{
        switch(format0) {
          case 0:
            prach_pdu->prach_format = 0;
            break;
          case 1:
            prach_pdu->prach_format = 1;
            break;
          case 2:
            prach_pdu->prach_format = 2;
            break;
          case 3:
            prach_pdu->prach_format = 3;
            break;
          case 0xa1:
            prach_pdu->prach_format = 4;
            break;
          case 0xa2:
            prach_pdu->prach_format = 5;
            break;
          case 0xa3:
            prach_pdu->prach_format = 6;
            break;
          case 0xb1:
            prach_pdu->prach_format = 7;
            break;
          case 0xb4:
            prach_pdu->prach_format = 8;
            break;
          case 0xc0:
            prach_pdu->prach_format = 9;
            break;
          case 0xc2:
            prach_pdu->prach_format = 10;
            break;
        default:
          AssertFatal(1==0,"Invalid PRACH format");
        }
      }		
     }
    }
   }
  }
  }
}

void nr_schedule_msg2(uint16_t rach_frame, uint16_t rach_slot,
                      uint16_t *msg2_frame, uint16_t *msg2_slot,
                      NR_ServingCellConfigCommon_t *scc,
                      uint16_t monitoring_slot_period,
                      uint16_t monitoring_offset,uint8_t index,uint8_t num_active_ssb){

  // preferentially we schedule the msg2 in the mixed slot or in the last dl slot
  // if they are allowed by search space configuration

  uint8_t mu = *scc->ssbSubcarrierSpacing;
  uint8_t response_window = scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->rach_ConfigGeneric.ra_ResponseWindow;
  uint8_t slot_window;
  // number of mixed slot or of last dl slot if there is no mixed slot
  uint8_t last_dl_slot_period = scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofDownlinkSlots;
  // lenght of tdd period in slots
  uint8_t tdd_period_slot =  scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofDownlinkSlots + scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofUplinkSlots;
  if (scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofDownlinkSymbols == 0)
    last_dl_slot_period--;
  if ((scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofDownlinkSymbols > 0) || (scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofUplinkSymbols > 0))
    tdd_period_slot++;

  // computing start of next period
  uint8_t start_next_period = (rach_slot-(rach_slot%tdd_period_slot)+tdd_period_slot)%nr_slots_per_frame[mu];
  *msg2_slot = start_next_period + last_dl_slot_period; // initializing scheduling of slot to next mixed (or last dl) slot
  *msg2_frame = (*msg2_slot>(rach_slot))? rach_frame : (rach_frame +1);
 
  switch(response_window){
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl1:
      slot_window = 1;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl2:
      slot_window = 2;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl4:
      slot_window = 4;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl8:
      slot_window = 8;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl10:
      slot_window = 10;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl20:
      slot_window = 20;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl40:
      slot_window = 40;
      break;
    case NR_RACH_ConfigGeneric__ra_ResponseWindow_sl80:
      slot_window = 80;
      break;
    default:
      AssertFatal(1==0,"Invalid response window value %d\n",response_window);
  }
  AssertFatal(slot_window<=nr_slots_per_frame[mu],"Msg2 response window needs to be lower or equal to 10ms");

  // slot and frame limit to transmit msg2 according to response window
  uint8_t slot_limit = (rach_slot + slot_window)%nr_slots_per_frame[mu];
  //uint8_t frame_limit = (slot_limit>(rach_slot))? rach_frame : (rach_frame +1);


  // go to previous slot if the current scheduled slot is beyond the response window
  // and if the slot is not among the PDCCH monitored ones (38.213 10.1)
  while ((*msg2_slot>slot_limit) || ((*msg2_frame*nr_slots_per_frame[mu]+*msg2_slot-monitoring_offset)%monitoring_slot_period !=0))  {
    if((*msg2_slot%tdd_period_slot) > 0)
      (*msg2_slot)--;
    else
      AssertFatal(1==0,"No available DL slot to schedule msg2 has been found");
  }
}


void nr_initiate_ra_proc(module_id_t module_idP,
                         int CC_id,
                         frame_t frameP,
                         sub_frame_t slotP,
                         uint16_t preamble_index,
                         uint8_t freq_index,
                         uint8_t symbol,
                         int16_t timing_offset){

  uint8_t ul_carrier_id = 0; // 0 for NUL 1 for SUL
  NR_SearchSpace_t *ss;
  // ra_rnti from 5.1.3 in 38.321
  uint16_t ra_rnti=1+symbol+(slotP*14)+(freq_index*14*80)+(ul_carrier_id*14*80*8);

  uint16_t msg2_frame, msg2_slot,monitoring_slot_period,monitoring_offset;
  gNB_MAC_INST *nr_mac = RC.nrmac[module_idP];
  NR_COMMON_channels_t *cc = &nr_mac->common_channels[CC_id];
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  NR_RA_t *ra = &cc->ra[0];
  // if the preamble received correspond to one of the listed
  // the UE sent a RACH either for starting RA procedure or RA procedure failed and UE retries
  int pr_found=0;
  for (int i = 0; i < ra->preambles.num_preambles; i++) {
    if (preamble_index == ra->preambles.preamble_list[i]) {
      pr_found=1;
      break;
    }
  }
  if (!pr_found) {
    LOG_E(MAC, "[gNB %d][RAPROC] FAILURE: preamble %d does not correspond to any of the ones in rach_ConfigDedicated\n",
          module_idP, preamble_index);
    return; // if the PRACH preamble does not correspond to any of the ones sent through RRC abort RA proc
  }
  // This should be handled differently when we use the initialBWP for RA
  ra->bwp_id=1;
  NR_BWP_Downlink_t *bwp=ra->secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.array[ra->bwp_id-1];

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_INITIATE_RA_PROC, 1);

  LOG_I(MAC, "[gNB %d][RAPROC] CC_id %d Frame %d, Slot %d  Initiating RA procedure for preamble index %d\n", module_idP, CC_id, frameP, slotP, preamble_index);

  if (ra->state == RA_IDLE) {

    uint8_t beam_index = ssb_index_from_prach(module_idP,
		                              frameP,
					      slotP,
					      preamble_index,
					      freq_index,
					      symbol);
    int loop = 0;
    LOG_D(MAC, "Frame %d, Slot %d: Activating RA process \n", frameP, slotP);
    ra->state = Msg2;
    ra->timing_offset = timing_offset;
    ra->preamble_slot = slotP;

    struct NR_PDCCH_ConfigCommon__commonSearchSpaceList *commonSearchSpaceList = bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->commonSearchSpaceList;
    AssertFatal(commonSearchSpaceList->list.count>0,
	        "common SearchSpace list has 0 elements\n");
    // Common searchspace list
    for (int i=0;i<commonSearchSpaceList->list.count;i++) {
      ss=commonSearchSpaceList->list.array[i];
      if(ss->searchSpaceId == *bwp->bwp_Common->pdcch_ConfigCommon->choice.setup->ra_SearchSpace)
        ra->ra_ss=ss;
    }

    // retrieving ra pdcch monitoring period and offset
    find_monitoring_periodicity_offset_common(ra->ra_ss,
                                              &monitoring_slot_period,
                                              &monitoring_offset);

    nr_schedule_msg2(frameP, slotP, &msg2_frame, &msg2_slot, scc, monitoring_slot_period, monitoring_offset,beam_index,cc->num_active_ssb);

    ra->Msg2_frame = msg2_frame;
    ra->Msg2_slot = msg2_slot;

    LOG_I(MAC, "%s() Msg2[%04d%d] SFN/SF:%04d%d\n", __FUNCTION__, ra->Msg2_frame, ra->Msg2_slot, frameP, slotP);
    if (!ra->cfra) {
      do {
        ra->rnti = (taus() % 65518) + 1;
        loop++;
      }
      while (loop != 100 && !((find_nr_UE_id(module_idP, ra->rnti) == -1) && (find_nr_RA_id(module_idP, CC_id, ra->rnti) == -1) && ra->rnti >= 1 && ra->rnti <= 65519));
      if (loop == 100) {
        LOG_E(MAC,"%s:%d:%s: [RAPROC] initialisation random access aborted\n", __FILE__, __LINE__, __FUNCTION__);
        abort();
      }
    }
    ra->RA_rnti = ra_rnti;
    ra->preamble_index = preamble_index;
    ra->beam_id = beam_index;

    LOG_I(MAC,"[gNB %d][RAPROC] CC_id %d Frame %d Activating Msg2 generation in frame %d, slot %d using RA rnti %x SSB index %u\n",
      module_idP,
      CC_id,
      frameP,
      ra->Msg2_frame,
      ra->Msg2_slot,
      ra->RA_rnti,
      cc->ssb_index[beam_index]);

    return;
  }
  LOG_E(MAC, "[gNB %d][RAPROC] FAILURE: CC_id %d Frame %d initiating RA procedure for preamble index %d\n", module_idP, CC_id, frameP, preamble_index);

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_INITIATE_RA_PROC, 0);
}

void nr_schedule_RA(module_id_t module_idP, frame_t frameP, sub_frame_t slotP, int num_slots_per_tdd){

  //uint8_t i = 0;
  int CC_id = 0;
  gNB_MAC_INST *mac = RC.nrmac[module_idP];
  NR_COMMON_channels_t *cc = &mac->common_channels[CC_id];

  start_meas(&mac->schedule_ra);

//  for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
//    for (int i = 0; i < NR_NB_RA_PROC_MAX; i++) {
  
//	NR_RA_t *ra = &cc->ra[i];
	NR_RA_t *ra = &cc->ra[0];
  if (ra->state != IDLE)
      LOG_I(MAC,"RA[state:%d], frame %d %d\n",ra->state, frameP, slotP);
  switch (ra->state){
    case Msg2:
      nr_generate_Msg2(module_idP, CC_id, frameP, slotP);
      break;
    case Msg4:
      if (ra->Msg4_frame == frameP && ra->Msg4_slot == slotP )
          nr_generate_Msg4(module_idP, CC_id, frameP, slotP, num_slots_per_tdd, ra);
      break;
    case WAIT_Msg4_ACK:
      //check_Msg4_retransmission(module_idP, CC_id, frameP, slotP);
          nr_check_Msg4_Ack(module_idP, CC_id, frameP, slotP, ra);
      break;
    default:
    break;
  }
//    }
//  }
  stop_meas(&mac->schedule_ra);
}

void nr_get_Msg3alloc(NR_ServingCellConfigCommon_t *scc,
                      NR_BWP_Uplink_t *ubwp,
                      sub_frame_t current_slot,
                      frame_t current_frame,
                      NR_RA_t *ra) {

  // msg3 is schedulend in mixed slot in the following TDD period
  // for now we consider a TBS of 18 bytes

  int mu = ubwp->bwp_Common->genericParameters.subcarrierSpacing;
  int StartSymbolIndex, NrOfSymbols, startSymbolAndLength, temp_slot;
  ra->Msg3_tda_id = 16; // initialization to a value above limit

  for (int i=0; i<ubwp->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList->list.count; i++) {
    startSymbolAndLength = ubwp->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList->list.array[i]->startSymbolAndLength;
    SLIV2SL(startSymbolAndLength, &StartSymbolIndex, &NrOfSymbols);
    // we want to transmit in the uplink symbols of mixed slot
    if (NrOfSymbols == scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofUplinkSymbols) {
      ra->Msg3_tda_id = i;
      break;
    }
  }
  AssertFatal(ra->Msg3_tda_id<16,"Unable to find Msg3 time domain allocation in list\n");

  uint8_t k2 = *ubwp->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList->list.array[ra->Msg3_tda_id]->k2;

  temp_slot = current_slot + k2 + DELTA[mu]; // msg3 slot according to 8.3 in 38.213
  ra->Msg3_slot = temp_slot%nr_slots_per_frame[mu];
  if (nr_slots_per_frame[mu]>temp_slot)
    ra->Msg3_frame = current_frame;
  else
    ra->Msg3_frame = current_frame + (temp_slot/nr_slots_per_frame[mu]);

  LOG_I(MAC, "[RAPROC] Msg3 slot %d: current slot %u Msg3 frame %u k2 %u Msg3_tda_id %u start symbol index %u\n", ra->Msg3_slot, current_slot, ra->Msg3_frame, k2,ra->Msg3_tda_id, StartSymbolIndex);
  ra->msg3_nb_rb = 18;
  ra->msg3_first_rb = 0;
}


void nr_schedule_reception_msg3(module_id_t module_idP, int CC_id, frame_t frameP, sub_frame_t slotP){
  gNB_MAC_INST                                *mac = RC.nrmac[module_idP];
  nfapi_nr_ul_tti_request_t                   *ul_req = &mac->UL_tti_req[0];
  NR_COMMON_channels_t                        *cc = &mac->common_channels[CC_id];
  NR_RA_t                                     *ra = &cc->ra[0];

  if (ra->state == WAIT_Msg3) {
    if ((frameP == ra->Msg3_frame) && (slotP == ra->Msg3_slot) ){
      ul_req->SFN = ra->Msg3_frame;
      ul_req->Slot = ra->Msg3_slot;
      ul_req->pdus_list[ul_req->n_pdus].pdu_type = NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE;
      ul_req->pdus_list[ul_req->n_pdus].pdu_size = sizeof(nfapi_nr_pusch_pdu_t);
      ul_req->pdus_list[ul_req->n_pdus].pusch_pdu = ra->pusch_pdu;
      ul_req->n_pdus+=1;
    }
  }
}

void nr_add_msg3(module_id_t module_idP, int CC_id, frame_t frameP, sub_frame_t slotP){

  gNB_MAC_INST                                   *mac = RC.nrmac[module_idP];
  NR_COMMON_channels_t                            *cc = &mac->common_channels[CC_id];
  NR_ServingCellConfigCommon_t                   *scc = cc->ServingCellConfigCommon;
  NR_RA_t                                         *ra = &cc->ra[0];

  if (ra->state == RA_IDLE) {
    LOG_W(MAC,"RA is not active for RA %X. skipping msg3 scheduling\n", ra->rnti);
    return;
  }

  LOG_I(MAC, "[gNB %d][RAPROC] Frame %d, Subframe %d : CC_id %d RA is active, Msg3 in (%d,%d)\n", module_idP, frameP, slotP, CC_id, ra->Msg3_frame, ra->Msg3_slot);

  nfapi_nr_pusch_pdu_t  *pusch_pdu = &ra->pusch_pdu;
  memset(pusch_pdu, 0, sizeof(nfapi_nr_pusch_pdu_t));

  AssertFatal(ra->secondaryCellGroup,
              "no secondaryCellGroup for RNTI %04x\n",
              ra->crnti);
  AssertFatal(ra->secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count == 1,
    "downlinkBWP_ToAddModList has %d BWP!\n", ra->secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count);
  NR_BWP_Uplink_t *ubwp = ra->secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->uplinkBWP_ToAddModList->list.array[ra->bwp_id - 1];
  LOG_D(MAC, "Frame %d, Subframe %d Adding Msg3 UL Config Request for (%d,%d) : (%d,%d,%d) for rnti: %d\n",
    frameP,
    slotP,
    ra->Msg3_frame,
    ra->Msg3_slot,
    ra->msg3_nb_rb,
    ra->msg3_first_rb,
    ra->msg3_round,
    ra->rnti);

  int startSymbolAndLength = ubwp->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList->list.array[ra->Msg3_tda_id]->startSymbolAndLength;
  int start_symbol_index,nr_of_symbols;
  SLIV2SL(startSymbolAndLength, &start_symbol_index, &nr_of_symbols);

  pusch_pdu->pdu_bit_map = PUSCH_PDU_BITMAP_PUSCH_DATA;
  pusch_pdu->rnti = ra->rnti;
  pusch_pdu->handle = 0;
  int abwp_size  = NRRIV2BW(ubwp->bwp_Common->genericParameters.locationAndBandwidth,275);
  int abwp_start = NRRIV2PRBOFFSET(ubwp->bwp_Common->genericParameters.locationAndBandwidth,275);
  int ibwp_size  = NRRIV2BW(scc->uplinkConfigCommon->initialUplinkBWP->genericParameters.locationAndBandwidth,275);
  int ibwp_start = NRRIV2PRBOFFSET(scc->uplinkConfigCommon->initialUplinkBWP->genericParameters.locationAndBandwidth,275);
  if ((ibwp_start < abwp_start) || (ibwp_size > abwp_size))
    pusch_pdu->bwp_start = abwp_start;
  else
    pusch_pdu->bwp_start = ibwp_start;
  pusch_pdu->bwp_size = ibwp_size;
  pusch_pdu->subcarrier_spacing = ubwp->bwp_Common->genericParameters.subcarrierSpacing;
  pusch_pdu->cyclic_prefix = 0;
  pusch_pdu->mcs_index = 0;
  pusch_pdu->mcs_table = 0;
  pusch_pdu->target_code_rate = nr_get_code_rate_ul(pusch_pdu->mcs_index,pusch_pdu->mcs_table);
  pusch_pdu->qam_mod_order = nr_get_Qm_ul(pusch_pdu->mcs_index,pusch_pdu->mcs_table);
  if (scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg3_transformPrecoder == NULL)
    pusch_pdu->transform_precoding = 1;
  else
    pusch_pdu->transform_precoding = 0;
  pusch_pdu->data_scrambling_id = *scc->physCellId;
  pusch_pdu->nrOfLayers = 1;
  pusch_pdu->ul_dmrs_symb_pos = 1<<start_symbol_index; // ok for now but use fill dmrs mask later
  pusch_pdu->dmrs_config_type = 0;
  pusch_pdu->ul_dmrs_scrambling_id = *scc->physCellId; //If provided and the PUSCH is not a msg3 PUSCH, otherwise, L2 should set this to physical cell id.
  pusch_pdu->scid = 0; //DMRS sequence initialization [TS38.211, sec 6.4.1.1.1]. Should match what is sent in DCI 0_1, otherwise set to 0.
  pusch_pdu->dmrs_ports = 1;  // 6.2.2 in 38.214 only port 0 to be used
  pusch_pdu->num_dmrs_cdm_grps_no_data = 2;  // no data in dmrs symbols as in 6.2.2 in 38.214
  pusch_pdu->resource_alloc = 1; //type 1
  pusch_pdu->rb_start = ra->msg3_first_rb + ibwp_start - abwp_start; // as for 6.3.1.7 in 38.211
  if (ra->msg3_nb_rb > pusch_pdu->bwp_size)
    AssertFatal(1==0,"MSG3 allocated number of RBs exceed the BWP size\n");
  else
    pusch_pdu->rb_size = ra->msg3_nb_rb;
  pusch_pdu->vrb_to_prb_mapping = 0;
  if (ubwp->bwp_Dedicated->pusch_Config->choice.setup->frequencyHopping == NULL)
    pusch_pdu->frequency_hopping = 0;
  else
    pusch_pdu->frequency_hopping = 1;
  //pusch_pdu->tx_direct_current_location;//The uplink Tx Direct Current location for the carrier. Only values in the value range of this field between 0 and 3299, which indicate the subcarrier index within the carrier corresponding 1o the numerology of the corresponding uplink BWP and value 3300, which indicates "Outside the carrier" and value 3301, which indicates "Undetermined position within the carrier" are used. [TS38.331, UplinkTxDirectCurrentBWP IE]
  pusch_pdu->uplink_frequency_shift_7p5khz = 0;
  //Resource Allocation in time domain
  pusch_pdu->start_symbol_index = start_symbol_index;
  pusch_pdu->nr_of_symbols = nr_of_symbols;
  //Optional Data only included if indicated in pduBitmap
  pusch_pdu->pusch_data.rv_index = 0;  // 8.3 in 38.213
  pusch_pdu->pusch_data.harq_process_id = 0;
  pusch_pdu->pusch_data.new_data_indicator = 1; // new data
  pusch_pdu->pusch_data.num_cb = 0;
  pusch_pdu->pusch_data.tb_size = nr_compute_tbs(pusch_pdu->qam_mod_order,
                                                 pusch_pdu->target_code_rate,
                                                 pusch_pdu->rb_size,
                                                 pusch_pdu->nr_of_symbols,
                                                 12, // nb dmrs set for no data in dmrs symbol
                                                 0, //nb_rb_oh
                                                 0, // to verify tb scaling
                                                 pusch_pdu->nrOfLayers = 1)>>3;

  // calling function to fill rar message
  nr_fill_rar(module_idP, ra, cc->RAR_pdu.payload, pusch_pdu);

#if 1 //#ifdef LOG_PUSCH_PARAMES
static int log_first = 0;
 if (log_first == 0)
  {
    log_first = 1;
  LOG_I(MAC, "pusch configure: rnti %d, bwpsize %d, bwpstart %d, subcarrier_spacing %d  cyclic_prefix %d\n",
  pusch_pdu->rnti,
  pusch_pdu->bwp_size,
  pusch_pdu->bwp_start,
  pusch_pdu->subcarrier_spacing,
  pusch_pdu->cyclic_prefix);
  LOG_I(MAC, "pusch configure: target_code_rate %d, qam_mod_order %d, mcs_index %d, mcs_table %d , transform_precoding %d, data_scrambling_id %d\n",
  pusch_pdu->target_code_rate,
  pusch_pdu->qam_mod_order,
  pusch_pdu->mcs_index,
  pusch_pdu->mcs_table,
  pusch_pdu->transform_precoding,
  pusch_pdu->data_scrambling_id);
  LOG_I(MAC, "pusch configure: nrOfLayers %d, ul_dmrs_symb_pos %d dmrs_config_type %d, ul_dmrs_scrambling_id %d, scid %d, num_dmrs_cdm_grps_no_data %d\n",
  pusch_pdu->nrOfLayers,
  pusch_pdu->ul_dmrs_symb_pos,
  pusch_pdu->dmrs_config_type,
  pusch_pdu->ul_dmrs_scrambling_id,
  pusch_pdu->scid,
  pusch_pdu->num_dmrs_cdm_grps_no_data);
  LOG_I(MAC, "pusch configure: dmrs_ports %d, resource_alloc %d, rb_start %d, rb_size %d, vrb_to_prb_mapping %d, frequency_hopping %d, tx_direct_current_location %d, uplink_frequency_shift_7p5khz %d, start_symbol_index %d, nr_of_symbols %d\n",
  pusch_pdu->dmrs_ports,
  pusch_pdu->resource_alloc,
  pusch_pdu->rb_start,
  pusch_pdu->rb_size,
  pusch_pdu->vrb_to_prb_mapping,
  pusch_pdu->frequency_hopping,
  pusch_pdu->tx_direct_current_location,
  pusch_pdu->uplink_frequency_shift_7p5khz,
  pusch_pdu->start_symbol_index,
  pusch_pdu->nr_of_symbols);
  }
#endif


}

// WIP
// todo:
// - fix me
// - get msg3 alloc (see nr_process_rar)
void nr_generate_Msg2(module_id_t module_idP,
                      int CC_id,
                      frame_t frameP,
                      sub_frame_t slotP)
{

  int dci_formats[2], rnti_types[2], mcsIndex;
  int startSymbolAndLength = 0, StartSymbolIndex = -1, NrOfSymbols = 14, StartSymbolIndex_tmp, NrOfSymbols_tmp, x_Overhead, time_domain_assignment = 0;
  gNB_MAC_INST                      *nr_mac = RC.nrmac[module_idP];
  NR_COMMON_channels_t                  *cc = &nr_mac->common_channels[CC_id];
  NR_RA_t                               *ra = &cc->ra[CC_id];
  NR_SearchSpace_t *ss = ra->ra_ss;

  uint16_t RA_rnti = ra->RA_rnti;
  long locationAndBandwidth;

  // check if UE is doing RA on CORESET0 , InitialBWP or configured BWP from SCD
  // get the BW of the PDCCH for PDCCH size and RAR PDSCH size

  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  int dci10_bw;

  if (ra->coreset0_configured == 1) {
    AssertFatal(1==0,"This is a standalone condition\n");
  }
  else { // on configured BWP or initial LDBWP, bandwidth parameters in DCI correspond size of initialBWP
    locationAndBandwidth = scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters.locationAndBandwidth;
    dci10_bw = NRRIV2BW(locationAndBandwidth,275); 
  }

  if ((ra->Msg2_frame == frameP) && (ra->Msg2_slot == slotP)) {

    nfapi_nr_dl_tti_request_body_t *dl_req = &nr_mac->DL_req[CC_id].dl_tti_request_body;
    nfapi_nr_pdu_t *tx_req = &nr_mac->TX_req[CC_id].pdu_list[nr_mac->TX_req[CC_id].Number_of_PDUs];

    nfapi_nr_dl_tti_request_pdu_t *dl_tti_pdcch_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs];
    memset((void*)dl_tti_pdcch_pdu,0,sizeof(nfapi_nr_dl_tti_request_pdu_t));
    dl_tti_pdcch_pdu->PDUType = NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE;
    dl_tti_pdcch_pdu->PDUSize = (uint8_t)(2+sizeof(nfapi_nr_dl_tti_pdcch_pdu));

    nfapi_nr_dl_tti_request_pdu_t *dl_tti_pdsch_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs+1];
    memset((void *)dl_tti_pdsch_pdu,0,sizeof(nfapi_nr_dl_tti_request_pdu_t));
    dl_tti_pdsch_pdu->PDUType = NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE;
    dl_tti_pdsch_pdu->PDUSize = (uint8_t)(2+sizeof(nfapi_nr_dl_tti_pdsch_pdu));

    nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu_rel15 = &dl_tti_pdcch_pdu->pdcch_pdu.pdcch_pdu_rel15;
    nfapi_nr_dl_tti_pdsch_pdu_rel15_t *pdsch_pdu_rel15 = &dl_tti_pdsch_pdu->pdsch_pdu.pdsch_pdu_rel15;

    // Checking if the DCI allocation is feasible in current subframe
    if (dl_req->nPDUs == NFAPI_NR_MAX_DL_TTI_PDUS) {
      LOG_I(MAC, "[RAPROC] Subframe %d: FAPI DL structure is full, skip scheduling UE %d\n", slotP, RA_rnti);
      return;
    }

    LOG_I(MAC,"[gNB %d] [RAPROC] CC_id %d Frame %d, slotP %d: Generating RAR DCI, state %d\n", module_idP, CC_id, frameP, slotP, ra->state);

    // This code from this point on will not work on initialBWP or CORESET0
    AssertFatal(ra->bwp_id>0,"cannot work on initialBWP for now\n");

    AssertFatal(ra->secondaryCellGroup,
                "no secondaryCellGroup for RNTI %04x\n",
                ra->crnti);
    AssertFatal(ra->secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count == 1,
      "downlinkBWP_ToAddModList has %d BWP!\n", ra->secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count);
    NR_BWP_Downlink_t *bwp = ra->secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.array[ra->bwp_id - 1];
    NR_BWP_Uplink_t *ubwp=ra->secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->uplinkBWP_ToAddModList->list.array[ra->bwp_id-1];

    LOG_I(MAC, "[RAPROC] Scheduling common search space DCI type 1 dlBWP BW %d\n", dci10_bw);

    // Qm>2 not allowed for RAR
    if (get_softmodem_params()->do_ra)
      mcsIndex = 9;
    else
      mcsIndex = 0;
  
    pdsch_pdu_rel15->pduBitmap = 0;
    pdsch_pdu_rel15->rnti = RA_rnti;
    pdsch_pdu_rel15->pduIndex = 0;


    pdsch_pdu_rel15->BWPSize  = NRRIV2BW(bwp->bwp_Common->genericParameters.locationAndBandwidth,275);
    pdsch_pdu_rel15->BWPStart = NRRIV2PRBOFFSET(bwp->bwp_Common->genericParameters.locationAndBandwidth,275);
    pdsch_pdu_rel15->SubcarrierSpacing = bwp->bwp_Common->genericParameters.subcarrierSpacing;
    pdsch_pdu_rel15->CyclicPrefix = 0;
    pdsch_pdu_rel15->NrOfCodewords = 1;
    pdsch_pdu_rel15->targetCodeRate[0] = nr_get_code_rate_dl(mcsIndex,0);
    pdsch_pdu_rel15->qamModOrder[0] = 2;
    pdsch_pdu_rel15->mcsIndex[0] = mcsIndex;
    if (bwp->bwp_Dedicated->pdsch_Config->choice.setup->mcs_Table == NULL)
      pdsch_pdu_rel15->mcsTable[0] = 0;
    else{
      if (*bwp->bwp_Dedicated->pdsch_Config->choice.setup->mcs_Table == 0)
        pdsch_pdu_rel15->mcsTable[0] = 1;
      else
        pdsch_pdu_rel15->mcsTable[0] = 2;
    }
    pdsch_pdu_rel15->rvIndex[0] = 0;
    pdsch_pdu_rel15->dataScramblingId = *scc->physCellId;
    pdsch_pdu_rel15->nrOfLayers = 1;
    pdsch_pdu_rel15->transmissionScheme = 0;
    pdsch_pdu_rel15->refPoint = 0;
    pdsch_pdu_rel15->dmrsConfigType = 0;
    pdsch_pdu_rel15->dlDmrsScramblingId = *scc->physCellId;
    pdsch_pdu_rel15->SCID = 0;
    pdsch_pdu_rel15->numDmrsCdmGrpsNoData = 2;
    pdsch_pdu_rel15->dmrsPorts = 1;
    pdsch_pdu_rel15->resourceAlloc = 1;
    pdsch_pdu_rel15->rbStart = 0;
    pdsch_pdu_rel15->rbSize = 16;
    pdsch_pdu_rel15->VRBtoPRBMapping = 0; // non interleaved

    for (int i=0; i<bwp->bwp_Common->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.count; i++) {
      startSymbolAndLength = bwp->bwp_Common->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.array[i]->startSymbolAndLength;
      SLIV2SL(startSymbolAndLength, &StartSymbolIndex_tmp, &NrOfSymbols_tmp);
      if (NrOfSymbols_tmp < NrOfSymbols) {
        NrOfSymbols = NrOfSymbols_tmp;
        StartSymbolIndex = StartSymbolIndex_tmp;
        time_domain_assignment = i; // this is short PDSCH added to the config to fit mixed slot
      }
    }

    AssertFatal(StartSymbolIndex >= 0, "StartSymbolIndex is negative\n");

    pdsch_pdu_rel15->StartSymbolIndex = StartSymbolIndex;
    pdsch_pdu_rel15->NrOfSymbols      = NrOfSymbols;
    pdsch_pdu_rel15->dlDmrsSymbPos = fill_dmrs_mask(NULL, scc->dmrs_TypeA_Position, NrOfSymbols);

    dci_pdu_rel15_t dci_pdu_rel15[MAX_DCI_CORESET];
    dci_pdu_rel15[0].frequency_domain_assignment.val = PRBalloc_to_locationandbandwidth0(pdsch_pdu_rel15->rbSize,
										     pdsch_pdu_rel15->rbStart,dci10_bw);
    dci_pdu_rel15[0].time_domain_assignment.val = time_domain_assignment;
    dci_pdu_rel15[0].vrb_to_prb_mapping.val = 0;
    dci_pdu_rel15[0].mcs = pdsch_pdu_rel15->mcsIndex[0];
    dci_pdu_rel15[0].tb_scaling = 0;

    LOG_I(MAC, "[RAPROC] DCI type 1 payload: freq_alloc %d (%d,%d,%d), time_alloc %d, vrb to prb %d, mcs %d tb_scaling %d \n",
	  dci_pdu_rel15[0].frequency_domain_assignment.val,
	  pdsch_pdu_rel15->rbStart,
	  pdsch_pdu_rel15->rbSize,
	  dci10_bw,
	  dci_pdu_rel15[0].time_domain_assignment.val,
	  dci_pdu_rel15[0].vrb_to_prb_mapping.val,
	  dci_pdu_rel15[0].mcs,
	  dci_pdu_rel15[0].tb_scaling);

    uint8_t nr_of_candidates, aggregation_level;
    find_aggregation_candidates(&aggregation_level, &nr_of_candidates, ss);
    NR_ControlResourceSet_t *coreset = get_coreset(bwp, ss, 0 /* common */);
    int CCEIndex = allocate_nr_CCEs(nr_mac,
                                    bwp,
                                    coreset,
                                    aggregation_level,
                                    0, // Y
                                    0, // m
                                    nr_of_candidates);

    if (CCEIndex < 0) {
      LOG_E(MAC, "%s(): cannot find free CCE for RA RNTI %04x!\n", __func__, ra->rnti);
      return;
    }
    nr_configure_pdcch(nr_mac,
                       pdcch_pdu_rel15,
                       RA_rnti,
                       ss,
                       coreset,
                       scc,
                       bwp,
                       aggregation_level,
                       CCEIndex);

    LOG_I(MAC, "Frame %d: Subframe %d : Adding common DL DCI for RA_RNTI %x\n", frameP, slotP, RA_rnti);

    dci_formats[0] = NR_DL_DCI_FORMAT_1_0;
    rnti_types[0] = NR_RNTI_RA;

    LOG_I(MAC, "[RAPROC] DCI params: rnti %d, rnti_type %d, dci_format %d coreset params: FreqDomainResource %llx, start_symbol %d  n_symb %d\n",
      pdcch_pdu_rel15->dci_pdu.RNTI[0],
      rnti_types[0],
      dci_formats[0],
      (unsigned long long)pdcch_pdu_rel15->FreqDomainResource,
      pdcch_pdu_rel15->StartSymbolIndex,
      pdcch_pdu_rel15->DurationSymbols);

    fill_dci_pdu_rel15(scc,ra->secondaryCellGroup,pdcch_pdu_rel15, &dci_pdu_rel15[0], dci_formats, rnti_types,dci10_bw,ra->bwp_id);

    dl_req->nPDUs+=2;

    // Program UL processing for Msg3
    nr_get_Msg3alloc(scc, ubwp, slotP, frameP, ra);
    LOG_I(MAC, "Frame %d, Subframe %d: Setting Msg3 reception for Frame %d Subframe %d\n", frameP, slotP, ra->Msg3_frame, ra->Msg3_slot);
    nr_add_msg3(module_idP, CC_id, frameP, slotP);
    ra->state = WAIT_Msg3;
    LOG_I(MAC,"[gNB %d][RAPROC] Frame %d, Subframe %d: RA state %d\n", module_idP, frameP, slotP, ra->state);

    x_Overhead = 0;
    nr_get_tbs_dl(&dl_tti_pdsch_pdu->pdsch_pdu, x_Overhead, pdsch_pdu_rel15->numDmrsCdmGrpsNoData, dci_pdu_rel15[0].tb_scaling);

    // DL TX request
    tx_req->PDU_length = pdsch_pdu_rel15->TBSize[0];
    tx_req->PDU_index = nr_mac->pdu_index[CC_id]++;
    tx_req->num_TLV = 1;
    tx_req->TLVs[0].length = 8;
    nr_mac->TX_req[CC_id].SFN = frameP;
    nr_mac->TX_req[CC_id].Number_of_PDUs++;
    nr_mac->TX_req[CC_id].Slot = slotP;
    memcpy((void*)&tx_req->TLVs[0].value.direct[0], (void*)&cc[CC_id].RAR_pdu.payload[0], tx_req->TLVs[0].length);

    /* mark the corresponding RBs as used */
    uint8_t *vrb_map = cc[CC_id].vrb_map;
    for (int rb = 0; rb < pdsch_pdu_rel15->rbSize; rb++)
      vrb_map[rb + pdsch_pdu_rel15->rbStart] = 1;


#if 1 // LOG_PDCCH_PARAMES
  static int log_first = 0;
    if(log_first == 0)
    {
        //log_first =1;
        LOG_I(MAC, "NB PDCCH PARAMS: rnti %d, BWPSize %d, BWPStart %d, SubcarrierSpacing %d, CCE %d, L %d dci_length %d dci_format %d dci index %d\n",
              pdcch_pdu_rel15->dci_pdu.RNTI[0],
              pdcch_pdu_rel15->BWPSize,
              pdcch_pdu_rel15->BWPStart,
              pdcch_pdu_rel15->SubcarrierSpacing,
              pdcch_pdu_rel15->dci_pdu.CceIndex[pdcch_pdu_rel15->numDlDci-1],
              pdcch_pdu_rel15->dci_pdu.AggregationLevel[pdcch_pdu_rel15->numDlDci-1],
              pdcch_pdu_rel15->dci_pdu.PayloadSizeBits[pdcch_pdu_rel15->numDlDci-1],
              dci_formats[0], pdcch_pdu_rel15->numDlDci-1);
        LOG_I(MAC, "NB PDCCH PARAMS: coreset:frequency_domain_resource %d %d %d %d %d %d\n", pdcch_pdu_rel15->FreqDomainResource[0],
                      pdcch_pdu_rel15->FreqDomainResource[1],
                      pdcch_pdu_rel15->FreqDomainResource[2],
                      pdcch_pdu_rel15->FreqDomainResource[3],
                      pdcch_pdu_rel15->FreqDomainResource[4],
                      pdcch_pdu_rel15->FreqDomainResource[5]);
        LOG_I(MAC, "NB PDCCH PARAMS: coreset:StartSymbolIndex %d duration %d CceRegMappingType %d RegBundleSize %d InterleaverSize %d ShiftIndex %d CoreSetType %d precoder_granularity %d, pdcch_dmrs_scrambling_id %d,scrambling_rnti %d\n",
              pdcch_pdu_rel15->StartSymbolIndex,
              pdcch_pdu_rel15->DurationSymbols,
              pdcch_pdu_rel15->CceRegMappingType,
              pdcch_pdu_rel15->RegBundleSize,
              pdcch_pdu_rel15->InterleaverSize,
              pdcch_pdu_rel15->ShiftIndex,
              pdcch_pdu_rel15->CoreSetType,
              pdcch_pdu_rel15->precoderGranularity,
              pdcch_pdu_rel15->dci_pdu.ScramblingId[0],
              pdcch_pdu_rel15->dci_pdu.ScramblingRNTI[0]);
        
    }
#endif

#if 1 // LOG_PDSCH_PARAMES
   static int log_first_pdsch_ra = 0;
   if (log_first_pdsch_ra == 0)
   {
      //log_first_pdsch_ra = 1;
      LOG_I(MAC, "NB PDSCH PARAMS: RA rnti %d, bwp (%d, %d), scs %d, codewords %d, coderate %d, mod %d, mcs (%d, %d), rv %d, dataScramId %d, layers %d, tm %d, refPoint %d \n ",
            pdsch_pdu_rel15->rnti,
            pdsch_pdu_rel15->BWPSize,
            pdsch_pdu_rel15->BWPStart,
            pdsch_pdu_rel15->SubcarrierSpacing,
            pdsch_pdu_rel15->NrOfCodewords,
            pdsch_pdu_rel15->targetCodeRate[0],
            pdsch_pdu_rel15->qamModOrder[0],
            pdsch_pdu_rel15->mcsIndex[0],
            pdsch_pdu_rel15->mcsTable[0],
            pdsch_pdu_rel15->rvIndex[0],
            pdsch_pdu_rel15->dataScramblingId,
            pdsch_pdu_rel15->nrOfLayers,
            pdsch_pdu_rel15->transmissionScheme,
            pdsch_pdu_rel15->refPoint      
            );
      LOG_I(MAC, "NB PDSCH PARAMS: RA dlDmrsScramblingId %d, scid %d, numDmrsCdmGrpsNoData %d, dmrsPorts %d, resourceAlloc %d, rb (%d, %d), symb (%d, %d), dmrsType %d, dmrsPos %d, vrb2prb %d\n",
            pdsch_pdu_rel15->dlDmrsScramblingId,
            pdsch_pdu_rel15->SCID,
            pdsch_pdu_rel15->numDmrsCdmGrpsNoData,
            pdsch_pdu_rel15->dmrsPorts,
            pdsch_pdu_rel15->resourceAlloc,
            pdsch_pdu_rel15->rbStart,
            pdsch_pdu_rel15->rbSize,
            pdsch_pdu_rel15->StartSymbolIndex,
            pdsch_pdu_rel15->NrOfSymbols,
            pdsch_pdu_rel15->dmrsConfigType,
            pdsch_pdu_rel15->dlDmrsSymbPos,
            pdsch_pdu_rel15->VRBtoPRBMapping 
            );
   }
#endif

  }
}

void nr_simple_dlsch_preprocessor_msg4(module_id_t module_id,
                                  frame_t frame,
                                  sub_frame_t slot,
                                  int num_slots_per_tdd,
                                  NR_RA_t *ra,
                                  int UE_id) {
  const int CC_id = 0;                                  
  gNB_MAC_INST                      *nr_mac = RC.nrmac[module_id];
  NR_COMMON_channels_t                  *cc = &nr_mac->common_channels[CC_id];
  NR_SearchSpace_t *ss = ra->ra_ss;
                                  
  NR_UE_info_t *UE_info = &RC.nrmac[module_id]->UE_info;
  int16_t rrc_sdu_length = 0;

  NR_UE_sched_ctrl_t *sched_ctrl = &UE_info->UE_sched_ctrl[UE_id];
  sched_ctrl->rbSize = 0;

  /* Retrieve amount of data to send for this UE */
  sched_ctrl->num_total_bytes = 0;
  
  /*
   rrc_sdu_length = mac_rrc_data_req(module_idP, CC_idP, frameP, CCCH,
                                        UE_RNTI(module_idP,UE_id),1,  // 1 transport block
                                        &cc[CC_idP].CCCH_pdu.payload[0], 0);  // not used in this case
  */
  rrc_sdu_length = 500;
  sched_ctrl->num_total_bytes += rrc_sdu_length;
  if (sched_ctrl->num_total_bytes == 0)
    return;
  LOG_D(MAC,
              "[gNB %d][RAPROC] CC_id %d Frame %d, slot %d: UE_id %d, rrc_sdu_length %d\n",
              module_id, CC_id, frame, slot, UE_id, rrc_sdu_length);

  /* Find a free CCE */
  sched_ctrl->search_space = ra->ra_ss;
  uint8_t nr_of_candidates;
  find_aggregation_candidates(&sched_ctrl->aggregation_level,
                              &nr_of_candidates,
                              sched_ctrl->search_space);
  sched_ctrl->coreset = get_coreset(
      sched_ctrl->active_bwp, sched_ctrl->search_space, 0 /* common */);
  int cid = sched_ctrl->coreset->controlResourceSetId;

  sched_ctrl->cce_index = allocate_nr_CCEs(RC.nrmac[module_id],
                                           sched_ctrl->active_bwp,
                                           sched_ctrl->coreset,
                                           sched_ctrl->aggregation_level,
                                           0,
                                           0,
                                           nr_of_candidates);
                                        
  if (sched_ctrl->cce_index < 0) {
    LOG_E(MAC, "%s(): could not find CCE for UE %d\n", __func__, UE_id);
    return;
  }
  UE_info->num_pdcch_cand[UE_id][cid]++;

  /* Find PUCCH occasion */
  nr_acknack_scheduling(module_id,
                        UE_id,
                        frame,
                        slot,
                        num_slots_per_tdd,
                        &sched_ctrl->pucch_sched_idx,
                        &sched_ctrl->pucch_occ_idx);

  AssertFatal(sched_ctrl->pucch_sched_idx >= 0, "no uplink slot for PUCCH found!\n");

  uint8_t *vrb_map = RC.nrmac[module_id]->common_channels[CC_id].vrb_map;
  sched_ctrl->current_harq_pid = slot % num_slots_per_tdd;
  const int current_harq_pid = sched_ctrl->current_harq_pid;
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
              ra->rnti);
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
    sched_ctrl->numDmrsCdmGrpsNoData = 2;

    // Freq-demain allocation
    while (rbStart < bwpSize && vrb_map[rbStart]) rbStart++;

    uint8_t N_PRB_DMRS =
        getN_PRB_DMRS(sched_ctrl->active_bwp, sched_ctrl->numDmrsCdmGrpsNoData);
    int nrOfSymbols = getNrOfSymbols(sched_ctrl->active_bwp,
                                     sched_ctrl->time_domain_allocation);

    int rbSize = 0;
    uint32_t TBS = 0;
    do {
      rbSize++;
      TBS = nr_compute_tbs(nr_get_Qm_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx),
                           nr_get_code_rate_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx),
                           rbSize,
                           nrOfSymbols,
                           N_PRB_DMRS, // FIXME // This should be multiplied by the
                                       // number of dmrs symbols
                           0 /* N_PRB_oh, 0 for initialBWP */,
                           0 /* tb_scaling */,
                           1 /* nrOfLayers */)
            >> 3;
    } while (rbStart + rbSize < bwpSize && !vrb_map[rbStart + rbSize] && TBS < sched_ctrl->num_total_bytes);
    sched_ctrl->rbSize = rbSize;
    sched_ctrl->rbStart = rbStart;
  }

  /* mark the corresponding RBs as used */
  for (int rb = 0; rb < sched_ctrl->rbSize; rb++)
    vrb_map[rb + sched_ctrl->rbStart] = 1;
}

void
nr_get_retransmission_timing(
                          frame_t *frameP,
                          sub_frame_t *subframeP)
//------------------------------------------------------------------------------
{
  *frameP = (*frameP + 1) % 1024;
  *subframeP = *subframeP;
  return;
}

void
nr_generate_Msg4(module_id_t module_id,
                 int CC_id,
                 frame_t frame,
                 sub_frame_t slot,
                 int num_slots_per_tdd,
                 NR_RA_t *ra) {
  /* PREPROCESSOR */

  NR_UE_info_t *UE_info = &RC.nrmac[module_id]->UE_info;
  //NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;
  int16_t rrc_sdu_length = 0;
  uint16_t msg4_padding = 0;
  uint16_t msg4_post_padding = 0;
  uint16_t msg4_header = 0;
  
  int UE_id = find_nr_UE_id_msg4(module_id, ra->rnti);

  if (UE_id < 0) {
    LOG_E(MAC, "Can't find UE for t-crnti %x, kill RA procedure for this UE\n",
          ra->rnti);
    nr_clear_ra_proc(module_id, CC_id, frame);
    return;
  }

  if (ra->coreset0_configured == 1) {
    AssertFatal(1==0,"This is a standalone condition\n");
  }
  else { // on configured BWP or initial LDBWP, bandwidth parameters in DCI correspond size of initialBWP
  
  }

  nr_simple_dlsch_preprocessor_msg4(module_id,
                                    frame,
                                    slot,
                                    num_slots_per_tdd,
                                    ra,
                                    UE_id);
  

  gNB_MAC_INST *gNB_mac = RC.nrmac[module_id];
   
  NR_UE_sched_ctrl_t *sched_ctrl = &UE_info->UE_sched_ctrl[UE_id];
    if (sched_ctrl->rbSize <= 0)
    {   // do nothing
         //continue;
     }
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
    const uint32_t TBS =
        nr_compute_tbs(nr_get_Qm_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx),
                       nr_get_code_rate_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx),
                       sched_ctrl->rbSize,
                       nrOfSymbols,
                       N_PRB_DMRS, // FIXME // This should be multiplied by the
                                   // number of dmrs symbols
                       0 /* N_PRB_oh, 0 for initialBWP */,
                       0 /* tb_scaling */,
                       1 /* nrOfLayers */)
        >> 3;

    const int current_harq_pid = sched_ctrl->current_harq_pid;
    NR_UE_harq_t *harq = &sched_ctrl->harq_processes[current_harq_pid];
    NR_sched_pucch *pucch = &sched_ctrl->sched_pucch[sched_ctrl->pucch_sched_idx][sched_ctrl->pucch_occ_idx];
    harq->feedback_slot = pucch->ul_slot;
    harq->is_waiting = 1;
    UE_info->mac_stats[UE_id].dlsch_rounds[harq->round]++;

    nfapi_nr_dl_tti_request_body_t *dl_req = &gNB_mac->DL_req[CC_id].dl_tti_request_body;
    nr_fill_nfapi_dl_pdu_common(module_id,
                         UE_id,
                         sched_ctrl->active_bwp->bwp_Id,
                         sched_ctrl->search_space,
                         sched_ctrl->coreset,
                         dl_req,
                         pucch,
                         1 /* nrOfLayers */,
                         sched_ctrl->mcs,
                         sched_ctrl->rbSize,
                         sched_ctrl->rbStart,
                         sched_ctrl->numDmrsCdmGrpsNoData,
                         getDmrsConfigType(sched_ctrl->active_bwp),
                         sched_ctrl->mcsTableIdx,
                         nr_get_code_rate_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx),
                         nr_get_Qm_dl(sched_ctrl->mcs, sched_ctrl->mcsTableIdx),
                         TBS,
                         sched_ctrl->time_domain_allocation,
                         startSymbolIndex,
                         nrOfSymbols,
                         sched_ctrl->aggregation_level,
                         sched_ctrl->cce_index,
                         current_harq_pid,
                         harq->ndi,
                         harq->round);

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
      LOG_W(MAC, "%d.%2d retransmission UE %d/RNTI %04x\n", frame, slot, UE_id, ra->rnti);
    } else { /* initial transmission */

      /* reserve space for timing advance of UE if necessary,
       * nr_generate_dlsch_pdu() checks for ta_apply and add TA CE if necessary */
      const int ta_len = (sched_ctrl->ta_apply) ? 2 : 0;

      /* Get RLC data TODO: remove random data retrieval */
      int header_length_total = 0;
      int header_length_last = 0;
      int sdu_length_total = 0;
      int num_sdus = 0;
      uint16_t sdu_lengths[NB_RB_MAX] = {0};
      uint8_t mac_sdus[MAX_NR_DLSCH_PAYLOAD_BYTES];
      unsigned char sdu_lcids[NB_RB_MAX] = {0};
      const int lcid = DL_SCH_LCID_DTCH;
      if (sched_ctrl->num_total_bytes > 0) {
        LOG_I(MAC,
              "[gNB %d][USER-PLANE DEFAULT DRB] Frame %d : DTCH->DLSCH, Requesting "
              "%d bytes from RLC (lcid %d total hdr len %d), TBS: %d \n \n",
              module_id,
              frame,
              TBS - ta_len - header_length_total - sdu_length_total - 3,
              lcid,
              header_length_total,
              TBS);
#if 0
        sdu_lengths[num_sdus] = mac_rrc_data_req(module_id, CC_idP, frameP, CCCH,
                                        UE_RNTI(module_idP,UE_id),1,  // 1 transport block
                                        &cc[CC_idP].CCCH_pdu.payload[0], 1);  


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
#else
        LOG_I(MAC, "Configuring DL_TX in %d.%d: random data\n", frame, slot);
        // fill dlsch_buffer with random data
        for (int i = 0; i < TBS; i++)
          mac_sdus[i] = (unsigned char) (lrand48()&0xff);
        sdu_lcids[0] = 0x3f; // DRB
        sdu_lengths[0] = TBS - ta_len - 3;
        header_length_total += 2 + (sdu_lengths[0] >= 128);
        sdu_length_total += sdu_lengths[0];
        num_sdus +=1;
#endif
        //ue_sched_ctl->uplane_inactivity_timer = 0;
      }
      else if (get_softmodem_params()->phy_test) {
        LOG_D(MAC, "Configuring DL_TX in %d.%d: random data\n", frame, slot);
        // fill dlsch_buffer with random data
        for (int i = 0; i < TBS; i++)
          mac_sdus[i] = (unsigned char) (lrand48()&0xff);
        sdu_lcids[0] = 0x3f; // DRB
        sdu_lengths[0] = TBS - ta_len - 3;
        header_length_total += 2 + (sdu_lengths[0] >= 128);
        sdu_length_total += sdu_lengths[0];
        num_sdus +=1;
      }

      UE_info->mac_stats[UE_id].dlsch_total_bytes += TBS;
      UE_info->mac_stats[UE_id].lc_bytes_tx[lcid] += sdu_length_total;

      // Check if there is data from RLC or CE
      const int post_padding = TBS >= 2 + header_length_total + sdu_length_total + ta_len;

      LOG_I(MAC, "Configuring DL_TX in %d.%d: TBS %d, header_length_total %d, sdu_length_total %d,ta_len %d, post_padding %d \n", frame, slot,
              TBS , header_length_total , sdu_length_total  ,ta_len  , post_padding  );
      // padding param currently not in use
      //padding = TBS - header_length_total - sdu_length_total - ta_len - 1;

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
          ra->cont_res_id, // contention res id
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
    nr_get_retransmission_timing(&ra->Msg4_frame, &ra->Msg4_slot);
    ra->state = WAIT_Msg4_ACK;
    LOG_I(MAC, "retrx time for msg4 is %d %d\n", ra->Msg4_frame, ra->Msg4_slot);
}

void
nr_check_Msg4_Ack(module_id_t module_id,
                  int CC_id,
                 frame_t frame,
                 sub_frame_t slot,
                 NR_RA_t *ra) {

  NR_UE_info_t *UE_info = &RC.nrmac[module_id]->UE_info;
  int UE_id = find_nr_UE_id_msg4(module_id, ra->rnti);
  NR_UE_sched_ctrl_t *sched_ctrl = &UE_info->UE_sched_ctrl[UE_id];
  const int current_harq_pid = sched_ctrl->current_harq_pid;
  NR_UE_harq_t *harq = &sched_ctrl->harq_processes[current_harq_pid];
  
  LOG_I(MAC, "ue %d, rnti %d, harq is waiting %d, round %d, frame %d %d\n", UE_id, ra->rnti, harq->is_waiting, harq->round, frame, slot);
  harq->is_waiting = 0;  // don't check the ack for the present

  if (harq->is_waiting == 0)
  {
      if ( harq->round == 0)
      {
          ra->state = IDLE;
          UE_info->active[UE_id] = true;
          LOG_I(MAC, "ue %d, rnti %d is active, frame %d %d\n", UE_id, ra->rnti, frame, slot);
      }
      else
      {
          ra->state = Msg4;
      }
  }  
}


void nr_clear_ra_proc(module_id_t module_idP, int CC_id, frame_t frameP){
  
  NR_RA_t *ra = &RC.nrmac[module_idP]->common_channels[CC_id].ra[0];
  LOG_D(MAC,"[gNB %d][RAPROC] CC_id %d Frame %d Clear Random access information rnti %x\n", module_idP, CC_id, frameP, ra->rnti);
  ra->state = IDLE;
  ra->timing_offset = 0;
  ra->RRC_timer = 20;
  ra->rnti = 0;
  ra->msg3_round = 0;
}


/////////////////////////////////////
//    Random Access Response PDU   //
//         TS 38.213 ch 8.2        //
//        TS 38.321 ch 6.2.3       //
/////////////////////////////////////
//| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |// bit-wise
//| E | T |       R A P I D       |//
//| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |//
//| R |           T A             |//
//|       T A         |  UL grant |//
//|            UL grant           |//
//|            UL grant           |//
//|            UL grant           |//
//|         T C - R N T I         |//
//|         T C - R N T I         |//
/////////////////////////////////////
//       UL grant  (27 bits)       //
/////////////////////////////////////
//| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |// bit-wise
//|-------------------|FHF|F_alloc|//
//|        Freq allocation        |//
//|    F_alloc    |Time allocation|//
//|      MCS      |     TPC   |CSI|//
/////////////////////////////////////
// WIP
// todo:
// - handle MAC RAR BI subheader
// - sending only 1 RAR subPDU
// - UL Grant: hardcoded CSI, TPC, time alloc
// - padding
void nr_fill_rar(uint8_t Mod_idP,
                 NR_RA_t * ra,
                 uint8_t * dlsch_buffer,
                 nfapi_nr_pusch_pdu_t  *pusch_pdu){

  LOG_I(MAC, "[gNB] Generate RAR MAC PDU frame %d slot %d preamble index %u", ra->Msg2_frame, ra-> Msg2_slot, ra->preamble_index);
  NR_RA_HEADER_RAPID *rarh = (NR_RA_HEADER_RAPID *) dlsch_buffer;
  NR_MAC_RAR *rar = (NR_MAC_RAR *) (dlsch_buffer + 1);
  unsigned char csi_req = 0, tpc_command;
  //uint8_t N_UL_Hop;
  uint8_t valid_bits;
  uint32_t ul_grant;
  uint16_t f_alloc, prb_alloc, bwp_size, truncation=0;

  tpc_command = 3; // this is 0 dB

  /// E/T/RAPID subheader ///
  // E = 0, one only RAR, first and last
  // T = 1, RAPID
  rarh->E = 0;
  rarh->T = 1;
  rarh->RAPID = ra->preamble_index;

  /// RAR MAC payload ///
  rar->R = 0;

  // TA command
  rar->TA1 = (uint8_t) (ra->timing_offset >> 5);    // 7 MSBs of timing advance
  rar->TA2 = (uint8_t) (ra->timing_offset & 0x1f);  // 5 LSBs of timing advance

  // TC-RNTI
  rar->TCRNTI_1 = (uint8_t) (ra->rnti >> 8);        // 8 MSBs of rnti
  rar->TCRNTI_2 = (uint8_t) (ra->rnti & 0xff);      // 8 LSBs of rnti

  // UL grant

  ra->msg3_TPC = tpc_command;

  bwp_size = pusch_pdu->bwp_size;
  prb_alloc = PRBalloc_to_locationandbandwidth0(ra->msg3_nb_rb, ra->msg3_first_rb, bwp_size);
  if (bwp_size>180) {
    AssertFatal(1==0,"Initial UBWP larger than 180 currently not supported");
  }
  else {
    valid_bits = (uint8_t)ceil(log2(bwp_size*(bwp_size+1)>>1));
  }

  if (pusch_pdu->frequency_hopping){
    AssertFatal(1==0,"PUSCH with frequency hopping currently not supported");
  } else {
    for (int i=0; i<valid_bits; i++)
      truncation |= (1<<i);
    f_alloc = (prb_alloc&truncation);
  }

  ul_grant = csi_req | (tpc_command << 1) | (pusch_pdu->mcs_index << 4) | (ra->Msg3_tda_id << 8) | (f_alloc << 12) | (pusch_pdu->frequency_hopping << 26);

  rar->UL_GRANT_1 = (uint8_t) (ul_grant >> 24) & 0x07;
  rar->UL_GRANT_2 = (uint8_t) (ul_grant >> 16) & 0xff;
  rar->UL_GRANT_3 = (uint8_t) (ul_grant >> 8) & 0xff;
  rar->UL_GRANT_4 = (uint8_t) ul_grant & 0xff;

}
