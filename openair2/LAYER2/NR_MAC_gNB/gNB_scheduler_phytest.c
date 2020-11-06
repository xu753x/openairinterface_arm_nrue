/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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

/*! \file gNB_scheduler_phytest.c
 * \brief gNB scheduling procedures in phy_test mode
 * \author  Guy De Souza, G. Casati
 * \date 07/2018
 * \email: desouza@eurecom.fr, guido.casati@iis.fraunhofer.de
 * \version 1.0
 * @ingroup _mac
 */

#include "nr_mac_gNB.h"
#include "SCHED_NR/sched_nr.h"
#include "NR_MAC_gNB/mac_proto.h"
#include "LAYER2/NR_MAC_COMMON/nr_mac_common.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "PHY/NR_TRANSPORT/nr_dci.h"
#include "executables/nr-softmodem.h"
#include "LAYER2/NR_MAC_COMMON/nr_mac.h"
#include "executables/softmodem-common.h"
#include "common/utils/nr/nr_common.h"
#include "NR_SCS-SpecificCarrier.h"
#include "NR_TDD-UL-DL-ConfigCommon.h"
#include "NR_FrequencyInfoUL.h"
#include "NR_RACH-ConfigGeneric.h"
#include "NR_RACH-ConfigCommon.h"
#include "NR_PUSCH-TimeDomainResourceAllocation.h"
#include "NR_PUSCH-ConfigCommon.h"
#include "NR_PUCCH-ConfigCommon.h"
#include "NR_PDSCH-TimeDomainResourceAllocation.h"
#include "NR_PDSCH-ConfigCommon.h"
#include "NR_RateMatchPattern.h"
#include "NR_RateMatchPatternLTE-CRS.h"
#include "NR_SearchSpace.h"
#include "NR_ControlResourceSet.h"

//#define UL_HARQ_PRINT
extern RAN_CONTEXT_t RC;

const uint8_t nr_rv_round_map[4] = {0, 2, 1, 3}; 
//#define ENABLE_MAC_PAYLOAD_DEBUG 1

//uint8_t mac_pdu[MAX_NR_DLSCH_PAYLOAD_BYTES];

/*Scheduling of DLSCH with associated DCI in common search space
 * current version has only a DCI for type 1 PDCCH for C_RNTI*/
void nr_schedule_css_dlsch_phytest(module_id_t   module_idP,
                                   frame_t       frameP,
                                   sub_frame_t   slotP) {
  uint8_t  CC_id;
  gNB_MAC_INST                      *nr_mac      = RC.nrmac[module_idP];
  NR_COMMON_channels_t              *cc = &nr_mac->common_channels[0];
  nfapi_nr_dl_tti_request_body_t    *dl_req;
  nfapi_nr_dl_tti_request_pdu_t     *dl_tti_pdcch_pdu;
  nfapi_nr_dl_tti_request_pdu_t     *dl_tti_pdsch_pdu;
  nfapi_nr_pdu_t        *TX_req;

  uint16_t rnti = 0x1234;
  
  //  int time_domain_assignment,k0;

  NR_ServingCellConfigCommon_t *scc=cc->ServingCellConfigCommon;

  int dlBWP_carrier_bandwidth = NRRIV2BW(scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters.locationAndBandwidth,275);

  
  /*
  int scs               = scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters.subcarrierSpacing;
  
  int slots_per_frame   = 10*(1<<scs);

  int FR                = *scc->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.array[0] >= 257 ? nr_FR2 : nr_FR1;
  */

  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    LOG_D(MAC, "Scheduling common search space DCI type 1 dlBWP BW.firstRB %d.%d\n",
	  dlBWP_carrier_bandwidth,
	  NRRIV2PRBOFFSET(scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters.locationAndBandwidth,275));
    
    
    dl_req = &nr_mac->DL_req[CC_id].dl_tti_request_body;
    dl_tti_pdcch_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs];
    memset((void*)dl_tti_pdcch_pdu,0,sizeof(nfapi_nr_dl_tti_request_pdu_t));
    dl_tti_pdcch_pdu->PDUType = NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE;
    dl_tti_pdcch_pdu->PDUSize = (uint8_t)(2+sizeof(nfapi_nr_dl_tti_pdcch_pdu));
    
    dl_tti_pdsch_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs+1];
    memset((void *)dl_tti_pdsch_pdu,0,sizeof(nfapi_nr_dl_tti_request_pdu_t));
    dl_tti_pdsch_pdu->PDUType = NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE;
    dl_tti_pdsch_pdu->PDUSize = (uint8_t)(2+sizeof(nfapi_nr_dl_tti_pdsch_pdu));

    
    //    nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu_rel15 = &dl_tti_pdcch_pdu->pdcch_pdu.pdcch_pdu_rel15;
    nfapi_nr_dl_tti_pdsch_pdu_rel15_t *pdsch_pdu_rel15 = &dl_tti_pdsch_pdu->pdsch_pdu.pdsch_pdu_rel15;
    
    pdsch_pdu_rel15->pduBitmap = 0;
    pdsch_pdu_rel15->rnti = rnti;
    pdsch_pdu_rel15->pduIndex = 0;

    // BWP
    pdsch_pdu_rel15->BWPSize  = NRRIV2BW(scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters.locationAndBandwidth,275);
    pdsch_pdu_rel15->BWPStart = NRRIV2PRBOFFSET(scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters.locationAndBandwidth,275);
    pdsch_pdu_rel15->SubcarrierSpacing = scc->downlinkConfigCommon->initialDownlinkBWP->genericParameters.subcarrierSpacing;
    pdsch_pdu_rel15->CyclicPrefix = 0;
    pdsch_pdu_rel15->NrOfCodewords = 1;
    int mcsIndex = 9;
    pdsch_pdu_rel15->targetCodeRate[0] = nr_get_code_rate_dl(mcsIndex,0);
    pdsch_pdu_rel15->qamModOrder[0] = 2;
    pdsch_pdu_rel15->mcsIndex[0] = mcsIndex;
    pdsch_pdu_rel15->mcsTable[0] = 0;
    pdsch_pdu_rel15->rvIndex[0] = 0;
    pdsch_pdu_rel15->dataScramblingId = *scc->physCellId;
    pdsch_pdu_rel15->nrOfLayers = 1;    
    pdsch_pdu_rel15->transmissionScheme = 0;
    pdsch_pdu_rel15->refPoint = 0; // Point A
    
    pdsch_pdu_rel15->dmrsConfigType = 0; // Type 1 by default for InitialBWP
    pdsch_pdu_rel15->dlDmrsScramblingId = *scc->physCellId;
    pdsch_pdu_rel15->SCID = 0;
    pdsch_pdu_rel15->numDmrsCdmGrpsNoData = 1;
    pdsch_pdu_rel15->dmrsPorts = 1;
    pdsch_pdu_rel15->resourceAlloc = 1;
    pdsch_pdu_rel15->rbStart = 0;
    pdsch_pdu_rel15->rbSize = 6;
    pdsch_pdu_rel15->VRBtoPRBMapping = 1; // non-interleaved, check if this is ok for initialBWP
    // choose shortest PDSCH
    int startSymbolAndLength=0;
    int StartSymbolIndex=-1,NrOfSymbols=14;
    int StartSymbolIndex_tmp,NrOfSymbols_tmp;

    for (int i=0;
	 i<scc->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.count;
	 i++) {
      startSymbolAndLength = scc->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.array[i]->startSymbolAndLength;
      SLIV2SL(startSymbolAndLength,&StartSymbolIndex_tmp,&NrOfSymbols_tmp);
      if (NrOfSymbols_tmp < NrOfSymbols) {
	NrOfSymbols = NrOfSymbols_tmp;
        StartSymbolIndex = StartSymbolIndex_tmp;
	//	k0 = *scc->downlinkConfigCommon->initialDownlinkBWP->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.array[i]->k0;
	//	time_domain_assignment = i;
      }
    }
    AssertFatal(StartSymbolIndex>=0,"StartSymbolIndex is negative\n");
    pdsch_pdu_rel15->StartSymbolIndex = StartSymbolIndex;
    pdsch_pdu_rel15->NrOfSymbols      = NrOfSymbols;
    pdsch_pdu_rel15->dlDmrsSymbPos = fill_dmrs_mask(NULL,
						    scc->dmrs_TypeA_Position,
						    NrOfSymbols);

    /*
    AssertFatal(k0==0,"k0 is not zero for Initial DL BWP TimeDomain Alloc\n");
    nr_configure_css_dci_initial(pdcch_pdu_rel15,
				 scs, 
				 scs, 
				 FR, 
				 0, 
				 0, 
				 0,
				 sfn_sf, slotP,
				 slots_per_frame,
				 dlBWP_carrier_bandwidth);
    
    
    pdu_rel15->frequency_domain_assignment = PRBalloc_to_locationandbandwidth0(pdsch_pdu_rel15->rbSize, 
                                                                               pdsch_pdu_rel15->rbStart, 
                                                                               dlBWP_carrier_bandwidth);
    pdu_rel15->time_domain_assignment = time_domain_assignment;
    
    pdu_rel15->vrb_to_prb_mapping = 1;
    pdu_rel15->mcs = 9;
    pdu_rel15->tb_scaling = 1;
    
    pdu_rel15->ra_preamble_index = 25;
    pdu_rel15->format_indicator = 1;
    pdu_rel15->ndi = 1;
    pdu_rel15->rv = 0;
    pdu_rel15->harq_pid = 0;
    pdu_rel15->dai = 2;
    pdu_rel15->tpc = 2;
    pdu_rel15->pucch_resource_indicator = 7;
    pdu_rel15->pdsch_to_harq_feedback_timing_indicator = 7;
    
    LOG_D(MAC, "[gNB scheduler phytest] DCI type 1 payload: freq_alloc %d, time_alloc %d, vrb to prb %d, mcs %d tb_scaling %d ndi %d rv %d\n",
	  pdu_rel15->frequency_domain_assignment,
	  pdu_rel15->time_domain_assignment,
	  pdu_rel15->vrb_to_prb_mapping,
	  pdu_rel15->mcs,
	  pdu_rel15->tb_scaling,
	  pdu_rel15->ndi,
	  pdu_rel15->rv);
    
    params_rel15->rnti = rnti;
    params_rel15->rnti_type = NFAPI_NR_RNTI_C;
    params_rel15->dci_format = NFAPI_NR_DL_DCI_FORMAT_1_0;
    //params_rel15->aggregation_level = 1;
    LOG_D(MAC, "DCI type 1 params: rnti %x, rnti_type %d, dci_format %d\n \
                coreset params: mux_pattern %d, n_rb %d, n_symb %d, rb_offset %d  \n \
                ss params : nb_ss_sets_per_slot %d, first symb %d, nb_slots %d, sfn_mod2 %d, first slot %d\n",
	  params_rel15->rnti,
	  params_rel15->rnti_type,
	  params_rel15->dci_format,
	  params_rel15->mux_pattern,
	  params_rel15->n_rb,
	  params_rel15->n_symb,
	  params_rel15->rb_offset,
	  params_rel15->nb_ss_sets_per_slot,
	  params_rel15->first_symbol,
	  params_rel15->nb_slots,
	  params_rel15->sfn_mod2,
	  params_rel15->first_slot);
    nr_get_tbs_dl(&dl_tti_pdsch_pdu->pdsch_pdu, dl_tti_dci_pdu->dci_dl_pdu,0);
    LOG_D(MAC, "DLSCH PDU: start PRB %d n_PRB %d start symbol %d nb_symbols %d nb_layers %d nb_codewords %d mcs %d\n",
	  pdsch_pdu_rel15->rbStart,
	  pdsch_pdu_rel15->rbSize,
	  pdsch_pdu_rel15->StartSymbolIndex,
	  pdsch_pdu_rel15->NrOfSymbols,
	  pdsch_pdu_rel15->nrOfLayers,
	  pdsch_pdu_rel15->NrOfCodewords,
	  pdsch_pdu_rel15->mcsIndex[0]);
    */
    
    dl_req->nPDUs+=2;
    
    TX_req = &nr_mac->TX_req[CC_id].pdu_list[nr_mac->TX_req[CC_id].Number_of_PDUs];
    TX_req->PDU_length = 6;
    TX_req->PDU_index = nr_mac->pdu_index[CC_id]++;
    TX_req->num_TLV = 1;
    TX_req->TLVs[0].length = 8;
    memcpy((void*)&TX_req->TLVs[0].value.direct[0],(void*)&cc[CC_id].RAR_pdu.payload[0],TX_req->TLVs[0].length);
    nr_mac->TX_req[CC_id].Number_of_PDUs++;
    nr_mac->TX_req[CC_id].SFN=frameP;
    nr_mac->TX_req[CC_id].Slot=slotP;
  }
}

int configure_fapi_dl_pdu_phytest(int Mod_idP,
                                  nfapi_nr_dl_tti_request_body_t *dl_req,
                                  NR_sched_pucch *pucch_sched,
                                  uint8_t *mcsIndex,
                                  uint16_t *rbSize,
                                  uint16_t *rbStart) {
  gNB_MAC_INST                        *nr_mac  = RC.nrmac[Mod_idP];
  NR_COMMON_channels_t                *cc      = nr_mac->common_channels;
  NR_ServingCellConfigCommon_t        *scc     = cc->ServingCellConfigCommon;

  nfapi_nr_dl_tti_request_pdu_t  *dl_tti_pdcch_pdu;
  nfapi_nr_dl_tti_request_pdu_t  *dl_tti_pdsch_pdu;

  int TBS;
  int bwp_id=1;
  int UE_id = 0;

  NR_UE_info_t *UE_info = &RC.nrmac[Mod_idP]->UE_info;

  NR_CellGroupConfig_t *secondaryCellGroup = UE_info->secondaryCellGroup[UE_id];
  AssertFatal(secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count == 1,
	      "downlinkBWP_ToAddModList has %d BWP!\n",
	      secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count);
  NR_BWP_Downlink_t *bwp=secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.array[bwp_id-1];

  AssertFatal(bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList!=NULL,"searchPsacesToAddModList is null\n");
  AssertFatal(bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList->list.count>0,
              "searchPsacesToAddModList is empty\n");

  dl_tti_pdcch_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs];
  memset((void*)dl_tti_pdcch_pdu,0,sizeof(nfapi_nr_dl_tti_request_pdu_t));
  dl_tti_pdcch_pdu->PDUType = NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE;
  dl_tti_pdcch_pdu->PDUSize = (uint8_t)(2+sizeof(nfapi_nr_dl_tti_pdcch_pdu));
  
  dl_tti_pdsch_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs+1];
  memset((void*)dl_tti_pdsch_pdu,0,sizeof(nfapi_nr_dl_tti_request_pdu_t));
  dl_tti_pdsch_pdu->PDUType = NFAPI_NR_DL_TTI_PDSCH_PDU_TYPE;
  dl_tti_pdsch_pdu->PDUSize = (uint8_t)(2+sizeof(nfapi_nr_dl_tti_pdsch_pdu));

  nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu_rel15 = &dl_tti_pdcch_pdu->pdcch_pdu.pdcch_pdu_rel15;
  nfapi_nr_dl_tti_pdsch_pdu_rel15_t *pdsch_pdu_rel15 = &dl_tti_pdsch_pdu->pdsch_pdu.pdsch_pdu_rel15;


  pdsch_pdu_rel15->pduBitmap = 0;
  pdsch_pdu_rel15->rnti = UE_info->rnti[UE_id];
  pdsch_pdu_rel15->pduIndex = 0;

  // BWP
  pdsch_pdu_rel15->BWPSize  = NRRIV2BW(bwp->bwp_Common->genericParameters.locationAndBandwidth,275);
  pdsch_pdu_rel15->BWPStart = NRRIV2PRBOFFSET(bwp->bwp_Common->genericParameters.locationAndBandwidth,275);
  pdsch_pdu_rel15->SubcarrierSpacing = bwp->bwp_Common->genericParameters.subcarrierSpacing;
  if (bwp->bwp_Common->genericParameters.cyclicPrefix) pdsch_pdu_rel15->CyclicPrefix = *bwp->bwp_Common->genericParameters.cyclicPrefix;
  else pdsch_pdu_rel15->CyclicPrefix=0;

  pdsch_pdu_rel15->NrOfCodewords = 1;
  int mcs = (mcsIndex!=NULL) ? *mcsIndex : 9;
  int current_harq_pid = UE_info->UE_sched_ctrl[UE_id].current_harq_pid;
  pdsch_pdu_rel15->targetCodeRate[0] = nr_get_code_rate_dl(mcs,0);
  pdsch_pdu_rel15->qamModOrder[0] = 2;
  pdsch_pdu_rel15->mcsIndex[0] = mcs;
  pdsch_pdu_rel15->mcsTable[0] = 0;
  pdsch_pdu_rel15->rvIndex[0] = nr_rv_round_map[UE_info->UE_sched_ctrl[UE_id].harq_processes[current_harq_pid].round];
  pdsch_pdu_rel15->dataScramblingId = *scc->physCellId;
  pdsch_pdu_rel15->nrOfLayers = 1;    
  pdsch_pdu_rel15->transmissionScheme = 0;
  pdsch_pdu_rel15->refPoint = 0; // Point A
  UE_info->mac_stats[UE_id].dlsch_rounds[UE_info->UE_sched_ctrl[UE_id].harq_processes[current_harq_pid].round]++;
  pdsch_pdu_rel15->dmrsConfigType = bwp->bwp_Dedicated->pdsch_Config->choice.setup->dmrs_DownlinkForPDSCH_MappingTypeA->choice.setup->dmrs_Type == NULL ? 0 : 1;  
  pdsch_pdu_rel15->dlDmrsScramblingId = *scc->physCellId;
  pdsch_pdu_rel15->SCID = 0;
  pdsch_pdu_rel15->numDmrsCdmGrpsNoData = 1;
  pdsch_pdu_rel15->dmrsPorts = 1;
  pdsch_pdu_rel15->resourceAlloc = 1;
  pdsch_pdu_rel15->rbStart = (rbStart!=NULL) ? *rbStart : 0;
  pdsch_pdu_rel15->rbSize = (rbSize!=NULL) ? *rbSize : pdsch_pdu_rel15->BWPSize;
  pdsch_pdu_rel15->VRBtoPRBMapping = 1; // non-interleaved, check if this is ok for initialBWP

  int startSymbolAndLength=0;
  int time_domain_assignment=2;
  int StartSymbolIndex,NrOfSymbols;

  AssertFatal(time_domain_assignment<bwp->bwp_Common->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.count,"time_domain_assignment %d>=%d\n",time_domain_assignment,bwp->bwp_Common->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.count);
  startSymbolAndLength = bwp->bwp_Common->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.array[time_domain_assignment]->startSymbolAndLength;
  SLIV2SL(startSymbolAndLength,&StartSymbolIndex,&NrOfSymbols);
  pdsch_pdu_rel15->StartSymbolIndex = StartSymbolIndex;
  pdsch_pdu_rel15->NrOfSymbols      = NrOfSymbols;
 
  //  k0 = *bwp->bwp_Common->pdsch_ConfigCommon->choice.setup->pdsch_TimeDomainAllocationList->list.array[i]->k0;
  pdsch_pdu_rel15->dlDmrsSymbPos    = fill_dmrs_mask(bwp->bwp_Dedicated->pdsch_Config->choice.setup,
						     scc->dmrs_TypeA_Position,
						     pdsch_pdu_rel15->NrOfSymbols);

  dci_pdu_rel15_t *dci_pdu_rel15 = calloc(MAX_DCI_CORESET,sizeof(dci_pdu_rel15_t));
  
  // bwp indicator
  int n_dl_bwp = secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count;
  if (n_dl_bwp < 4)
    dci_pdu_rel15[0].bwp_indicator.val = bwp_id;
  else
    dci_pdu_rel15[0].bwp_indicator.val = bwp_id - 1; // as per table 7.3.1.1.2-1 in 38.212
  // frequency domain assignment
  if (bwp->bwp_Dedicated->pdsch_Config->choice.setup->resourceAllocation==NR_PDSCH_Config__resourceAllocation_resourceAllocationType1)
    dci_pdu_rel15[0].frequency_domain_assignment.val = PRBalloc_to_locationandbandwidth0(pdsch_pdu_rel15->rbSize,
                                                                                         pdsch_pdu_rel15->rbStart,
										         NRRIV2BW(bwp->bwp_Common->genericParameters.locationAndBandwidth,275));
  else
    AssertFatal(1==0,"Only frequency resource allocation type 1 is currently supported\n");
  // time domain assignment
  dci_pdu_rel15[0].time_domain_assignment.val = time_domain_assignment; // row index used here instead of SLIV;
  // mcs and rv
  dci_pdu_rel15[0].mcs = pdsch_pdu_rel15->mcsIndex[0];
  dci_pdu_rel15[0].rv = pdsch_pdu_rel15->rvIndex[0];
  // harq pid and ndi
  dci_pdu_rel15[0].harq_pid = current_harq_pid;
  dci_pdu_rel15[0].ndi = UE_info->UE_sched_ctrl[UE_id].harq_processes[current_harq_pid].ndi;
  // DAI
  dci_pdu_rel15[0].dai[0].val = (pucch_sched->dai_c-1)&3;
  // TPC for PUCCH
  dci_pdu_rel15[0].tpc = UE_info->UE_sched_ctrl[UE_id].tpc1; // table 7.2.1-1 in 38.213
  // PUCCH resource indicator
  dci_pdu_rel15[0].pucch_resource_indicator = pucch_sched->resource_indicator;
  // PDSCH to HARQ TI
  dci_pdu_rel15[0].pdsch_to_harq_feedback_timing_indicator.val = pucch_sched->timing_indicator;
  UE_info->UE_sched_ctrl[UE_id].harq_processes[current_harq_pid].feedback_slot = pucch_sched->ul_slot;
  UE_info->UE_sched_ctrl[UE_id].harq_processes[current_harq_pid].is_waiting = 1;
  // antenna ports
  dci_pdu_rel15[0].antenna_ports.val = 0;  // nb of cdm groups w/o data 1 and dmrs port 0
  // dmrs sequence initialization
  dci_pdu_rel15[0].dmrs_sequence_initialization.val = pdsch_pdu_rel15->SCID;
  LOG_D(MAC, "[gNB scheduler phytest] DCI type 1 payload: freq_alloc %d (%d,%d,%d), time_alloc %d, vrb to prb %d, mcs %d tb_scaling %d ndi %d rv %d\n",
	dci_pdu_rel15[0].frequency_domain_assignment.val,
	pdsch_pdu_rel15->rbStart, 
	pdsch_pdu_rel15->rbSize,	
	NRRIV2BW(bwp->bwp_Common->genericParameters.locationAndBandwidth,275),
	dci_pdu_rel15[0].time_domain_assignment.val,
	dci_pdu_rel15[0].vrb_to_prb_mapping.val,
	dci_pdu_rel15[0].mcs,
	dci_pdu_rel15[0].tb_scaling,
	dci_pdu_rel15[0].ndi, 
	dci_pdu_rel15[0].rv);

  NR_SearchSpace_t *ss;
  int target_ss = NR_SearchSpace__searchSpaceType_PR_ue_Specific;

  AssertFatal(bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList!=NULL,"searchPsacesToAddModList is null\n");
  AssertFatal(bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList->list.count>0,
              "searchPsacesToAddModList is empty\n");

  int found=0;

  for (int i=0;i<bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList->list.count;i++) {
    ss=bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList->list.array[i];
    AssertFatal(ss->controlResourceSetId != NULL,"ss->controlResourceSetId is null\n");
    AssertFatal(ss->searchSpaceType != NULL,"ss->searchSpaceType is null\n");
    if (ss->searchSpaceType->present == target_ss) {
      found=1;
      break;
    }
  }
  AssertFatal(found==1,"Couldn't find an adequate searchspace\n");

  uint8_t nr_of_candidates, aggregation_level;
  find_aggregation_candidates(&aggregation_level, &nr_of_candidates, ss);
  NR_ControlResourceSet_t *coreset = get_coreset(bwp, ss, 1 /* dedicated */);
  const int cid = coreset->controlResourceSetId;
  const uint16_t Y = UE_info->Y[UE_id][cid][nr_mac->current_slot];
  const int m = UE_info->num_pdcch_cand[UE_id][cid];
  int CCEIndex = allocate_nr_CCEs(nr_mac,
                                  bwp,
                                  coreset,
                                  aggregation_level,
                                  Y,
                                  m,
                                  nr_of_candidates);
  if (CCEIndex < 0) {
    LOG_E(MAC, "%s(): CCE list not empty, couldn't schedule PDSCH\n", __func__);
    free(dci_pdu_rel15);
    return 0;
  }
  UE_info->num_pdcch_cand[UE_id][cid]++;

  nr_configure_pdcch(nr_mac,
                     pdcch_pdu_rel15,
                     UE_info->rnti[UE_id],
                     ss,
                     coreset,
                     scc,
                     bwp,
                     aggregation_level,
                     CCEIndex);

  int dci_formats[2];
  int rnti_types[2];
  
  if (ss->searchSpaceType->choice.ue_Specific->dci_Formats)
    dci_formats[0]  = NR_DL_DCI_FORMAT_1_1;
  else
    dci_formats[0]  = NR_DL_DCI_FORMAT_1_0;

  rnti_types[0]   = NR_RNTI_C;

  fill_dci_pdu_rel15(scc,secondaryCellGroup,pdcch_pdu_rel15,dci_pdu_rel15,dci_formats,rnti_types,pdsch_pdu_rel15->BWPSize,bwp_id);

  LOG_D(MAC, "DCI params: rnti %x, rnti_type %d, dci_format %d\n \
	                      coreset params: FreqDomainResource %llx, start_symbol %d  n_symb %d\n",
	pdcch_pdu_rel15->dci_pdu.RNTI[0],
	rnti_types[0],
	dci_formats[0],
	(unsigned long long)pdcch_pdu_rel15->FreqDomainResource,
	pdcch_pdu_rel15->StartSymbolIndex,
	pdcch_pdu_rel15->DurationSymbols);

  int x_Overhead = 0; // should be 0 for initialBWP
  nr_get_tbs_dl(&dl_tti_pdsch_pdu->pdsch_pdu,x_Overhead,pdsch_pdu_rel15->numDmrsCdmGrpsNoData,0);

  // Hardcode it for now
  TBS = dl_tti_pdsch_pdu->pdsch_pdu.pdsch_pdu_rel15.TBSize[0];
  if (UE_info->UE_sched_ctrl[UE_id].harq_processes[current_harq_pid].round==0)
    UE_info->mac_stats[UE_id].dlsch_total_bytes += TBS;

  LOG_D(MAC, "DLSCH PDU: start PRB %d n_PRB %d startSymbolAndLength %d start symbol %d nb_symbols %d nb_layers %d nb_codewords %d mcs %d TBS: %d\n",
	pdsch_pdu_rel15->rbStart,
	pdsch_pdu_rel15->rbSize,
	startSymbolAndLength,
	pdsch_pdu_rel15->StartSymbolIndex,
	pdsch_pdu_rel15->NrOfSymbols,
	pdsch_pdu_rel15->nrOfLayers,
	pdsch_pdu_rel15->NrOfCodewords,
	pdsch_pdu_rel15->mcsIndex[0],
	TBS);

  free(dci_pdu_rel15);
  return TBS; //Return TBS in bytes
}

void config_uldci(NR_BWP_Uplink_t *ubwp,
                  nfapi_nr_pusch_pdu_t *pusch_pdu,
                  nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu_rel15,
                  dci_pdu_rel15_t *dci_pdu_rel15,
                  int *dci_formats, int *rnti_types,
                  int time_domain_assignment, uint8_t tpc,
                  int n_ubwp, int bwp_id) {

  switch(dci_formats[(pdcch_pdu_rel15->numDlDci)-1]) {
    case NR_UL_DCI_FORMAT_0_0:
      dci_pdu_rel15->frequency_domain_assignment.val = PRBalloc_to_locationandbandwidth0(pusch_pdu->rb_size,
                                                                                         pusch_pdu->rb_start,
	                                                                                 NRRIV2BW(ubwp->bwp_Common->genericParameters.locationAndBandwidth,275));

      dci_pdu_rel15->time_domain_assignment.val = time_domain_assignment;
      dci_pdu_rel15->frequency_hopping_flag.val = pusch_pdu->frequency_hopping;
      dci_pdu_rel15->mcs = 9;

      dci_pdu_rel15->format_indicator = 0;
      dci_pdu_rel15->ndi = 1;
      dci_pdu_rel15->rv = 0;
      dci_pdu_rel15->harq_pid = 0;
      dci_pdu_rel15->tpc = 1;
      break;
    case NR_UL_DCI_FORMAT_0_1:
      dci_pdu_rel15->ndi = pusch_pdu->pusch_data.new_data_indicator;
      dci_pdu_rel15->rv = pusch_pdu->pusch_data.rv_index;
      dci_pdu_rel15->harq_pid = pusch_pdu->pusch_data.harq_process_id;
      dci_pdu_rel15->frequency_hopping_flag.val = pusch_pdu->frequency_hopping;
      dci_pdu_rel15->dai[0].val = 0; //TODO
      // bwp indicator
      if (n_ubwp < 4)
        dci_pdu_rel15->bwp_indicator.val = bwp_id;
      else
        dci_pdu_rel15->bwp_indicator.val = bwp_id - 1; // as per table 7.3.1.1.2-1 in 38.212
      // frequency domain assignment
      if (ubwp->bwp_Dedicated->pusch_Config->choice.setup->resourceAllocation==NR_PUSCH_Config__resourceAllocation_resourceAllocationType1)
        dci_pdu_rel15->frequency_domain_assignment.val = PRBalloc_to_locationandbandwidth0(pusch_pdu->rb_size,
                                                                                             pusch_pdu->rb_start,
                                                                                             NRRIV2BW(ubwp->bwp_Common->genericParameters.locationAndBandwidth,275));
      else
        AssertFatal(1==0,"Only frequency resource allocation type 1 is currently supported\n");
      // time domain assignment
      dci_pdu_rel15->time_domain_assignment.val = time_domain_assignment;
      // mcs
      dci_pdu_rel15->mcs = pusch_pdu->mcs_index;
      // tpc command for pusch
      dci_pdu_rel15->tpc = tpc;
      // SRS resource indicator
      if (ubwp->bwp_Dedicated->pusch_Config->choice.setup->txConfig != NULL) {
        if (*ubwp->bwp_Dedicated->pusch_Config->choice.setup->txConfig == NR_PUSCH_Config__txConfig_codebook)
          dci_pdu_rel15->srs_resource_indicator.val = 0; // taking resource 0 for SRS
        else
          AssertFatal(1==0,"Non Codebook configuration non supported\n");
      }
      // Antenna Ports
      dci_pdu_rel15->antenna_ports.val = 0; // TODO for now it is hardcoded, it should depends on cdm group no data and rank
      // DMRS sequence initialization
      dci_pdu_rel15->dmrs_sequence_initialization.val = pusch_pdu->scid;
      break;
    default :
      AssertFatal(1==0,"Valid UL formats are 0_0 and 0_1 \n");
  }

  LOG_D(MAC, "[gNB scheduler phytest] ULDCI type 0 payload: PDCCH CCEIndex %d, freq_alloc %d, time_alloc %d, freq_hop_flag %d, mcs %d tpc %d ndi %d rv %d\n",
	pdcch_pdu_rel15->dci_pdu.CceIndex[pdcch_pdu_rel15->numDlDci],
	dci_pdu_rel15->frequency_domain_assignment.val,
	dci_pdu_rel15->time_domain_assignment.val,
	dci_pdu_rel15->frequency_hopping_flag.val,
	dci_pdu_rel15->mcs,
	dci_pdu_rel15->tpc,
	dci_pdu_rel15->ndi, 
	dci_pdu_rel15->rv);

}
    

void configure_fapi_dl_Tx(module_id_t Mod_idP,
                          frame_t       frameP,
                          sub_frame_t   slotP,
                          nfapi_nr_dl_tti_request_body_t *dl_req,
                          nfapi_nr_pdu_t *tx_req,
                          int tbs_bytes,
                          int16_t pdu_index){

  int CC_id = 0;

  nfapi_nr_dl_tti_request_pdu_t  *dl_tti_pdsch_pdu = &dl_req->dl_tti_pdu_list[dl_req->nPDUs+1];
  nfapi_nr_dl_tti_pdsch_pdu_rel15_t *pdsch_pdu_rel15 = &dl_tti_pdsch_pdu->pdsch_pdu.pdsch_pdu_rel15;
  gNB_MAC_INST *nr_mac  = RC.nrmac[Mod_idP];

  LOG_D(MAC, "DLSCH PDU: start PRB %d n_PRB %d start symbol %d nb_symbols %d nb_layers %d nb_codewords %d mcs %d TBS (bytes): %d\n",
        pdsch_pdu_rel15->rbStart,
        pdsch_pdu_rel15->rbSize,
        pdsch_pdu_rel15->StartSymbolIndex,
        pdsch_pdu_rel15->NrOfSymbols,
        pdsch_pdu_rel15->nrOfLayers,
        pdsch_pdu_rel15->NrOfCodewords,
        pdsch_pdu_rel15->mcsIndex[0],
        tbs_bytes);

  dl_req->nPDUs+=2;

  tx_req->PDU_length = pdsch_pdu_rel15->TBSize[0];
  tx_req->PDU_index  = nr_mac->pdu_index[0]++;
  tx_req->num_TLV = 1;
  tx_req->TLVs[0].length = tbs_bytes +2;

  memcpy((void*)&tx_req->TLVs[0].value.direct[0], (void*)&nr_mac->UE_info.DLSCH_pdu[0][0].payload[0], tbs_bytes);

  nr_mac->TX_req[CC_id].Number_of_PDUs++;
  nr_mac->TX_req[CC_id].SFN = frameP;
  nr_mac->TX_req[CC_id].Slot = slotP;
}

void nr_schedule_uss_dlsch_phytest(module_id_t   module_idP,
                                   frame_t       frameP,
                                   sub_frame_t   slotP,
                                   NR_sched_pucch *pucch_sched,
                                   nfapi_nr_dl_tti_pdsch_pdu_rel15_t *dlsch_config){

  LOG_D(MAC, "In nr_schedule_uss_dlsch_phytest frame %d slot %d\n",frameP,slotP);

  int post_padding = 0, header_length_total = 0, sdu_length_total = 0, num_sdus = 0;
  int lcid, offset, i, header_length_last, TBS_bytes = 0;
  int UE_id = 0, CC_id = 0;

  gNB_MAC_INST *gNB_mac = RC.nrmac[module_idP];
  //NR_COMMON_channels_t                *cc           = nr_mac->common_channels;
  //NR_ServingCellConfigCommon_t *scc=cc->ServingCellConfigCommon;
  nfapi_nr_dl_tti_request_body_t *dl_req = &gNB_mac->DL_req[CC_id].dl_tti_request_body;
  nfapi_nr_pdu_t *tx_req = &gNB_mac->TX_req[CC_id].pdu_list[gNB_mac->TX_req[CC_id].Number_of_PDUs];

  mac_rlc_status_resp_t rlc_status;

  NR_UE_info_t *UE_info = &gNB_mac->UE_info;
 
  if (UE_info->num_UEs ==0) return;
 
  unsigned char sdu_lcids[NB_RB_MAX] = {0};
  uint16_t sdu_lengths[NB_RB_MAX] = {0};
  uint16_t rnti = UE_info->rnti[UE_id];
  NR_UE_sched_ctrl_t *ue_sched_ctl = &UE_info->UE_sched_ctrl[UE_id];

  uint8_t mac_sdus[MAX_NR_DLSCH_PAYLOAD_BYTES];
  
  LOG_D(MAC, "Scheduling UE specific search space DCI type 1\n");

  int ta_len = (ue_sched_ctl->ta_apply)?2:0;

  TBS_bytes = configure_fapi_dl_pdu_phytest(
      module_idP,
      dl_req,
      pucch_sched,
      dlsch_config != NULL ? dlsch_config->mcsIndex : NULL,
      dlsch_config != NULL ? &dlsch_config->rbSize : NULL,
      dlsch_config != NULL ? &dlsch_config->rbStart : NULL);

  if (TBS_bytes == 0)
   return;

  lcid = DL_SCH_LCID_DTCH;

  //for (lcid = NB_RB_MAX - 1; lcid >= DTCH; lcid--) {

  // TODO: check if the lcid is active

  LOG_D(MAC, "[gNB %d], Frame %d, DTCH%d->DLSCH, Checking RLC status (TBS %d bytes, len %d)\n",
      module_idP, frameP, lcid, TBS_bytes, TBS_bytes - ta_len - header_length_total - sdu_length_total - 3);

  //if (TBS_bytes - ta_len - header_length_total - sdu_length_total - 3 > 0) {
  rlc_status = mac_rlc_status_ind(module_idP,
      rnti,
      module_idP,
      frameP,
      slotP,
      ENB_FLAG_YES,
      MBMS_FLAG_NO,
      lcid,
      0,
      0);

  if (rlc_status.bytes_in_buffer > 0) {

    LOG_I(MAC, "configure fapi due to data availability \n");

    LOG_I(MAC, "[gNB %d][USER-PLANE DEFAULT DRB] Frame %d : DTCH->DLSCH, Requesting %d bytes from RLC (lcid %d total hdr len %d), TBS_bytes: %d \n \n",
        module_idP, frameP, TBS_bytes - ta_len - header_length_total - sdu_length_total - 3,
        lcid, header_length_total, TBS_bytes);

    sdu_lengths[num_sdus] = mac_rlc_data_req(module_idP,
        rnti,
        module_idP,
        frameP,
        ENB_FLAG_YES,
        MBMS_FLAG_NO,
        lcid,
        TBS_bytes - ta_len - header_length_total - sdu_length_total - 3,
        (char *)&mac_sdus[sdu_length_total],
        0,
        0);

    LOG_W(MAC, "[gNB %d][USER-PLANE DEFAULT DRB] Got %d bytes for DTCH %d \n", module_idP, sdu_lengths[num_sdus], lcid);

    sdu_lcids[num_sdus] = lcid;
    sdu_length_total += sdu_lengths[num_sdus];
    header_length_last = 1 + 1 + (sdu_lengths[num_sdus] >= 128);
    header_length_total += header_length_last;

    num_sdus++;

    //ue_sched_ctl->uplane_inactivity_timer = 0;
  }

  else {

    LOG_D(MAC,"Configuring DL_TX in %d.%d\n", frameP, slotP);

    // fill dlsch_buffer with random data
    for (i = 0; i < TBS_bytes; i++){
      mac_sdus[i] = (unsigned char) (lrand48()&0xff);
      //((uint8_t *)gNB_mac->UE_info.DLSCH_pdu[0][0].payload[0])[i] = (unsigned char) (lrand48()&0xff);
    }
    //Sending SDUs with size 1
    //Initialize elements of sdu_lcids and sdu_lengths
    sdu_lcids[0] = 0x3f; // DRB
    sdu_lengths[0] = TBS_bytes - ta_len - 3;
    header_length_total += 2 + (sdu_lengths[0] >= 128);
    sdu_length_total += sdu_lengths[0];
    num_sdus +=1;

    #if defined(ENABLE_MAC_PAYLOAD_DEBUG)
    if (frameP%100 == 0){
      LOG_I(MAC, "Printing first 10 payload bytes at the gNB side, Frame: %d, slot: %d, TBS size: %d \n", frameP, slotP, TBS_bytes);
      for(int i = 0; i < 10; i++) {
        LOG_I(MAC, "%x. ", ((uint8_t *)gNB_mac->UE_info.DLSCH_pdu[CC_id][0][0].payload[0])[i]);
      }
    }
    #endif

  }

  UE_info->mac_stats[UE_id].lc_bytes_tx[lcid] += sdu_length_total;

  // there is at least one SDU or TA command
  // if (num_sdus > 0 ){
  if (ta_len + sdu_length_total + header_length_total > 0) {

    // Check if there is data from RLC or CE
    if (TBS_bytes >= 2 + header_length_total + sdu_length_total + ta_len) {
      // we have to consider padding
      // padding param currently not in use
      //padding = TBS_bytes - header_length_total - sdu_length_total - ta_len - 1;
      post_padding = 1;
    } else {
      //padding = 0;
      post_padding = 0;
    }

    offset = nr_generate_dlsch_pdu(module_idP,
                                   &UE_info->UE_sched_ctrl[UE_id],
                                   (unsigned char *) mac_sdus,
                                   (unsigned char *) gNB_mac->UE_info.DLSCH_pdu[0][0].payload[0],
                                   num_sdus, //num_sdus
                                   sdu_lengths,
                                   sdu_lcids,
                                   255, // no drx
                                   NULL, // contention res id
                                   post_padding);

    // Padding: fill remainder of DLSCH with 0
    if (post_padding > 0){
      for (int j = 0; j < (TBS_bytes - offset); j++)
        gNB_mac->UE_info.DLSCH_pdu[0][0].payload[0][offset + j] = 0; // mac_pdu[offset + j] = 0;
    }

    configure_fapi_dl_Tx(module_idP, frameP, slotP, dl_req, tx_req, TBS_bytes, gNB_mac->pdu_index[CC_id]);

    if(IS_SOFTMODEM_NOS1){
      #if defined(ENABLE_MAC_PAYLOAD_DEBUG)
        LOG_I(MAC, "Printing first 10 payload bytes at the gNB side, Frame: %d, slot: %d, TBS size: %d \n \n", frameP, slotP, TBS_bytes);
        for(int i = 0; i < 10; i++) { // TBS_bytes dlsch_pdu_rel15->transport_block_size/8 6784/8
          LOG_I(MAC, "%x. ", mac_payload[i]);
        }
      #endif
    } else {
#if defined(ENABLE_MAC_PAYLOAD_DEBUG)
      if (frameP%100 == 0){
        LOG_I(MAC, "Printing first 10 payload bytes at the gNB side, Frame: %d, slot: %d, TBS size: %d \n", frameP, slotP, TBS_bytes);
        for(int i = 0; i < 10; i++) {
          LOG_I(MAC, "byte %d : %x\n", i,((uint8_t *)gNB_mac->UE_info.DLSCH_pdu[0][0].payload[0])[i]); //LOG_I(MAC, "%x. ", mac_payload[i]);
        }
      }
#endif
    }
  }
  else {  // There is no data from RLC or MAC header, so don't schedule
  }

}

int8_t select_ul_harq_pid(NR_UE_sched_ctrl_t *sched_ctrl) {

  uint8_t hrq_id;
  uint8_t max_ul_harq_pids = 3; // temp: for testing
  // schedule active harq processes
  NR_UE_ul_harq_t cur_harq;
  for (hrq_id=0; hrq_id < max_ul_harq_pids; hrq_id++) {
    cur_harq = sched_ctrl->ul_harq_processes[hrq_id];
    if (cur_harq.state==ACTIVE_NOT_SCHED) {
#ifdef UL_HARQ_PRINT
      printf("[SCHED] Found ulharq id %d, scheduling it for retransmission\n",hrq_id);
#endif
      return hrq_id;
    }
  }

  // schedule new harq processes
  for (hrq_id=0; hrq_id < max_ul_harq_pids; hrq_id++) {
    cur_harq = sched_ctrl->ul_harq_processes[hrq_id];
    if (cur_harq.state==INACTIVE) {
#ifdef UL_HARQ_PRINT
      printf("[SCHED] Found new ulharq id %d, scheduling it\n",hrq_id);
#endif
      return hrq_id;
    }
  }
  LOG_E(MAC,"All UL HARQ processes are busy. Cannot schedule ULSCH\n");
  return -1;
}

long get_K2(NR_BWP_Uplink_t *ubwp, int time_domain_assignment, int mu) {
  DevAssert(ubwp);
  const NR_PUSCH_TimeDomainResourceAllocation_t *tda_list = ubwp->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList->list.array[time_domain_assignment];
  if (tda_list->k2)
    return *tda_list->k2;
  else if (mu < 2)
    return 1;
  else if (mu == 2)
    return 2;
  else
    return 3;
}

void schedule_fapi_ul_pdu(int Mod_idP,
                          frame_t frameP,
                          sub_frame_t slotP,
                          int num_slots_per_tdd,
                          int ul_slots,
                          int time_domain_assignment,
                          uint64_t ulsch_in_slot_bitmap) {

  gNB_MAC_INST                      *nr_mac    = RC.nrmac[Mod_idP];
  NR_COMMON_channels_t                  *cc    = nr_mac->common_channels;
  NR_ServingCellConfigCommon_t         *scc    = cc->ServingCellConfigCommon;

  int bwp_id=1;
  int mu = scc->uplinkConfigCommon->initialUplinkBWP->genericParameters.subcarrierSpacing;
  int UE_id = 0;
  NR_UE_info_t *UE_info = &RC.nrmac[Mod_idP]->UE_info;
  AssertFatal(UE_info->active[UE_id],"Cannot find UE_id %d is not active\n",UE_id);

  NR_CellGroupConfig_t *secondaryCellGroup = UE_info->secondaryCellGroup[UE_id];
  AssertFatal(secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count == 1,
	      "downlinkBWP_ToAddModList has %d BWP!\n",
	      secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count);
  NR_BWP_Uplink_t *ubwp=secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->uplinkBWP_ToAddModList->list.array[bwp_id-1];
  int n_ubwp = secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->uplinkBWP_ToAddModList->list.count;
  NR_BWP_Downlink_t *bwp=secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.array[bwp_id-1];
  NR_PUSCH_Config_t *pusch_Config = ubwp->bwp_Dedicated->pusch_Config->choice.setup;

  AssertFatal(time_domain_assignment<ubwp->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList->list.count,
              "time_domain_assignment %d>=%d\n",time_domain_assignment,ubwp->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList->list.count);

  int K2 = get_K2(ubwp, time_domain_assignment,mu);
  /* check if slot is UL, and for phy test verify that it is in first TDD
   * period, slot 8 (for K2=2, this is at slot 6 in the gNB; because of UE
   * limitations).  Note that if K2 or the TDD configuration is changed, below
   * conditions might exclude each other and never be true */
  const int slot_idx = (slotP + K2) % num_slots_per_tdd;
  if (is_xlsch_in_slot(ulsch_in_slot_bitmap, slot_idx)
      && (!get_softmodem_params()->phy_test || slot_idx == 8)) {

    nfapi_nr_ul_dci_request_t *UL_dci_req = &RC.nrmac[Mod_idP]->UL_dci_req[0];
    UL_dci_req->SFN = frameP;
    UL_dci_req->Slot = slotP;
    nfapi_nr_ul_dci_request_pdus_t  *ul_dci_request_pdu;

    AssertFatal(bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList!=NULL,"searchPsacesToAddModList is null\n");
    AssertFatal(bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList->list.count>0,
                "searchPsacesToAddModList is empty\n");

    uint16_t rnti = UE_info->rnti[UE_id];

    int first_ul_slot = num_slots_per_tdd - ul_slots;
    NR_sched_pusch *pusch_sched = &UE_info->UE_sched_ctrl[UE_id].sched_pusch[slotP+K2-first_ul_slot];
    pusch_sched->frame = frameP;
    pusch_sched->slot = slotP + K2;
    pusch_sched->active = true;
    nfapi_nr_pusch_pdu_t  *pusch_pdu = &pusch_sched->pusch_pdu;
    memset(pusch_pdu,0,sizeof(nfapi_nr_pusch_pdu_t));

    LOG_I(MAC, "Scheduling UE specific PUSCH\n");
    //UL_tti_req = &nr_mac->UL_tti_req[CC_id];

    int dci_formats[2];
    int rnti_types[2];

    NR_SearchSpace_t *ss;
    int target_ss = NR_SearchSpace__searchSpaceType_PR_ue_Specific;

    AssertFatal(bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList!=NULL,"searchPsacesToAddModList is null\n");
    AssertFatal(bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList->list.count>0,
                "searchPsacesToAddModList is empty\n");

    int found=0;

    for (int i=0;i<bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList->list.count;i++) {
      ss=bwp->bwp_Dedicated->pdcch_Config->choice.setup->searchSpacesToAddModList->list.array[i];
      AssertFatal(ss->controlResourceSetId != NULL,"ss->controlResourceSetId is null\n");
      AssertFatal(ss->searchSpaceType != NULL,"ss->searchSpaceType is null\n");
      if (ss->searchSpaceType->present == target_ss) {
        found=1;
        break;
      }
    }
    AssertFatal(found==1,"Couldn't find an adequate searchspace\n");

    if (ss->searchSpaceType->choice.ue_Specific->dci_Formats)
      dci_formats[0]  = NR_UL_DCI_FORMAT_0_1;
    else
      dci_formats[0]  = NR_UL_DCI_FORMAT_0_0;

    rnti_types[0]   = NR_RNTI_C;

    //Resource Allocation in time domain
    int startSymbolAndLength=0;
    int StartSymbolIndex,NrOfSymbols,mapping_type;

    startSymbolAndLength = ubwp->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList->list.array[time_domain_assignment]->startSymbolAndLength;
    SLIV2SL(startSymbolAndLength,&StartSymbolIndex,&NrOfSymbols);
    pusch_pdu->start_symbol_index = StartSymbolIndex;
    pusch_pdu->nr_of_symbols = NrOfSymbols;

    mapping_type = ubwp->bwp_Common->pusch_ConfigCommon->choice.setup->pusch_TimeDomainAllocationList->list.array[time_domain_assignment]->mappingType;

    pusch_pdu->pdu_bit_map = PUSCH_PDU_BITMAP_PUSCH_DATA;
    pusch_pdu->rnti = rnti;
    pusch_pdu->handle = 0; //not yet used
  
    pusch_pdu->bwp_size  = NRRIV2BW(ubwp->bwp_Common->genericParameters.locationAndBandwidth,275);
    pusch_pdu->bwp_start = NRRIV2PRBOFFSET(ubwp->bwp_Common->genericParameters.locationAndBandwidth,275);
    pusch_pdu->subcarrier_spacing = ubwp->bwp_Common->genericParameters.subcarrierSpacing;
    pusch_pdu->cyclic_prefix = 0;

    if (pusch_Config->transformPrecoder == NULL) {
      if (scc->uplinkConfigCommon->initialUplinkBWP->rach_ConfigCommon->choice.setup->msg3_transformPrecoder == NULL)
        pusch_pdu->transform_precoding = 1;
      else
        pusch_pdu->transform_precoding = 0;
    }
    else
      pusch_pdu->transform_precoding = *pusch_Config->transformPrecoder;
    if (pusch_Config->dataScramblingIdentityPUSCH != NULL)
      pusch_pdu->data_scrambling_id = *pusch_Config->dataScramblingIdentityPUSCH;
    else
      pusch_pdu->data_scrambling_id = *scc->physCellId;

    pusch_pdu->mcs_index = 9;
    if (pusch_pdu->transform_precoding)
      pusch_pdu->mcs_table = get_pusch_mcs_table(pusch_Config->mcs_Table, 0,
                                                 dci_formats[0], rnti_types[0], target_ss, false);
    else
      pusch_pdu->mcs_table = get_pusch_mcs_table(pusch_Config->mcs_TableTransformPrecoder, 1,
                                                 dci_formats[0], rnti_types[0], target_ss, false);

    pusch_pdu->target_code_rate = nr_get_code_rate_ul(pusch_pdu->mcs_index,pusch_pdu->mcs_table);
    pusch_pdu->qam_mod_order = nr_get_Qm_ul(pusch_pdu->mcs_index,pusch_pdu->mcs_table);
    if (pusch_Config->tp_pi2BPSK!=NULL) {
      if(((pusch_pdu->mcs_table==3)&&(pusch_pdu->mcs_index<2)) ||
         ((pusch_pdu->mcs_table==4)&&(pusch_pdu->mcs_index<6))) {
        pusch_pdu->target_code_rate = pusch_pdu->target_code_rate>>1;
        pusch_pdu->qam_mod_order = pusch_pdu->qam_mod_order<<1;
      }
    }
    pusch_pdu->nrOfLayers = 1;

    //Pusch Allocation in frequency domain [TS38.214, sec 6.1.2.2]
    if (pusch_Config->resourceAllocation==NR_PUSCH_Config__resourceAllocation_resourceAllocationType1) {
      pusch_pdu->resource_alloc = 1; //type 1
      pusch_pdu->rb_start = 0;
      if (get_softmodem_params()->phy_test==1)
        pusch_pdu->rb_size = 50;
      else
        pusch_pdu->rb_size = pusch_pdu->bwp_size;
    }
    else
      AssertFatal(1==0,"Only frequency resource allocation type 1 is currently supported\n");

    pusch_pdu->vrb_to_prb_mapping = 0;

    if (pusch_Config->frequencyHopping==NULL)
      pusch_pdu->frequency_hopping = 0;
    else
      pusch_pdu->frequency_hopping = 1;

    //pusch_pdu->tx_direct_current_location;//The uplink Tx Direct Current location for the carrier. Only values in the value range of this field between 0 and 3299, which indicate the subcarrier index within the carrier corresponding 1o the numerology of the corresponding uplink BWP and value 3300, which indicates "Outside the carrier" and value 3301, which indicates "Undetermined position within the carrier" are used. [TS38.331, UplinkTxDirectCurrentBWP IE]
    //pusch_pdu->uplink_frequency_shift_7p5khz = 0;


    // --------------------
    // ------- DMRS -------
    // --------------------
    NR_DMRS_UplinkConfig_t *NR_DMRS_UplinkConfig;
    if (mapping_type == NR_PUSCH_TimeDomainResourceAllocation__mappingType_typeA)
      NR_DMRS_UplinkConfig = pusch_Config->dmrs_UplinkForPUSCH_MappingTypeA->choice.setup;
    else
      NR_DMRS_UplinkConfig = pusch_Config->dmrs_UplinkForPUSCH_MappingTypeB->choice.setup;
    if (NR_DMRS_UplinkConfig->dmrs_Type == NULL)
      pusch_pdu->dmrs_config_type = 0;
    else
      pusch_pdu->dmrs_config_type = 1;
    pusch_pdu->scid = 0;      // DMRS sequence initialization [TS38.211, sec 6.4.1.1.1]
    if (pusch_pdu->transform_precoding) { // transform precoding disabled
      long *scramblingid;
      if (pusch_pdu->scid == 0)
        scramblingid = NR_DMRS_UplinkConfig->transformPrecodingDisabled->scramblingID0;
      else
        scramblingid = NR_DMRS_UplinkConfig->transformPrecodingDisabled->scramblingID1;
      if (scramblingid == NULL)
        pusch_pdu->ul_dmrs_scrambling_id = *scc->physCellId;
      else
        pusch_pdu->ul_dmrs_scrambling_id = *scramblingid;
    }
    else {
      pusch_pdu->ul_dmrs_scrambling_id = *scc->physCellId;
      if (NR_DMRS_UplinkConfig->transformPrecodingEnabled->nPUSCH_Identity != NULL)
        pusch_pdu->pusch_identity = *NR_DMRS_UplinkConfig->transformPrecodingEnabled->nPUSCH_Identity;
      else
        pusch_pdu->pusch_identity = *scc->physCellId;
    }
    pusch_dmrs_AdditionalPosition_t additional_pos;
    if (NR_DMRS_UplinkConfig->dmrs_AdditionalPosition == NULL)
      additional_pos = 2;
    else {
      if (*NR_DMRS_UplinkConfig->dmrs_AdditionalPosition == NR_DMRS_UplinkConfig__dmrs_AdditionalPosition_pos3)
        additional_pos = 3;
      else
        additional_pos = *NR_DMRS_UplinkConfig->dmrs_AdditionalPosition;
    }
    pusch_maxLength_t pusch_maxLength;
    if (NR_DMRS_UplinkConfig->maxLength == NULL)
      pusch_maxLength = 1;
    else
      pusch_maxLength = 2;
    uint16_t l_prime_mask = get_l_prime(pusch_pdu->nr_of_symbols, mapping_type, additional_pos, pusch_maxLength);
    pusch_pdu->ul_dmrs_symb_pos = l_prime_mask << pusch_pdu->start_symbol_index;

    pusch_pdu->num_dmrs_cdm_grps_no_data = 1;
    pusch_pdu->dmrs_ports = 1;
    // --------------------------------------------------------------------------------------------------------------------------------------------

    // --------------------
    // ------- PTRS -------
    // --------------------
    if (NR_DMRS_UplinkConfig->phaseTrackingRS != NULL) {
      // TODO to be fixed from RRC config
      uint8_t ptrs_mcs1 = 2;  // higher layer parameter in PTRS-UplinkConfig
      uint8_t ptrs_mcs2 = 4;  // higher layer parameter in PTRS-UplinkConfig
      uint8_t ptrs_mcs3 = 10; // higher layer parameter in PTRS-UplinkConfig
      uint16_t n_rb0 = 25;    // higher layer parameter in PTRS-UplinkConfig
      uint16_t n_rb1 = 75;    // higher layer parameter in PTRS-UplinkConfig
      pusch_pdu->pusch_ptrs.ptrs_time_density = get_L_ptrs(ptrs_mcs1, ptrs_mcs2, ptrs_mcs3, pusch_pdu->mcs_index, pusch_pdu->mcs_table);
      pusch_pdu->pusch_ptrs.ptrs_freq_density = get_K_ptrs(n_rb0, n_rb1, pusch_pdu->rb_size);
      pusch_pdu->pusch_ptrs.ptrs_ports_list   = (nfapi_nr_ptrs_ports_t *) malloc(2*sizeof(nfapi_nr_ptrs_ports_t));
      pusch_pdu->pusch_ptrs.ptrs_ports_list[0].ptrs_re_offset = 0;

      pusch_pdu->pdu_bit_map |= PUSCH_PDU_BITMAP_PUSCH_PTRS; // enable PUSCH PTRS
    }
    else{
      pusch_pdu->pdu_bit_map &= ~PUSCH_PDU_BITMAP_PUSCH_PTRS; // disable PUSCH PTRS
    }

    // --------------------------------------------------------------------------------------------------------------------------------------------

    //Pusch Allocation in frequency domain [TS38.214, sec 6.1.2.2]
    //Optional Data only included if indicated in pduBitmap
    int8_t harq_id = select_ul_harq_pid(&UE_info->UE_sched_ctrl[UE_id]);
    if (harq_id < 0) return;
    NR_UE_ul_harq_t *cur_harq = &UE_info->UE_sched_ctrl[UE_id].ul_harq_processes[harq_id];
    pusch_pdu->pusch_data.harq_process_id = harq_id;
    pusch_pdu->pusch_data.new_data_indicator = cur_harq->ndi;
    pusch_pdu->pusch_data.rv_index = nr_rv_round_map[cur_harq->round];

    cur_harq->state = ACTIVE_SCHED;
    cur_harq->last_tx_slot = pusch_sched->slot;

    uint8_t num_dmrs_symb = 0;

    for(int dmrs_counter = pusch_pdu->start_symbol_index; dmrs_counter < pusch_pdu->start_symbol_index + pusch_pdu->nr_of_symbols; dmrs_counter++)
      num_dmrs_symb += ((pusch_pdu->ul_dmrs_symb_pos >> dmrs_counter) & 1);

    uint8_t N_PRB_DMRS;
    if (pusch_pdu->dmrs_config_type == 0) {
      N_PRB_DMRS = pusch_pdu->num_dmrs_cdm_grps_no_data*6;
    }
    else {
      N_PRB_DMRS = pusch_pdu->num_dmrs_cdm_grps_no_data*4;
    }

    pusch_pdu->pusch_data.tb_size = nr_compute_tbs(pusch_pdu->qam_mod_order,
                                                   pusch_pdu->target_code_rate,
                                                   pusch_pdu->rb_size,
                                                   pusch_pdu->nr_of_symbols,
                                                   N_PRB_DMRS * num_dmrs_symb,
                                                   0, //nb_rb_oh
                                                   0,
                                                   pusch_pdu->nrOfLayers)>>3;

    UE_info->mac_stats[UE_id].ulsch_rounds[cur_harq->round]++;
    if (cur_harq->round == 0) UE_info->mac_stats[UE_id].ulsch_total_bytes_scheduled+=pusch_pdu->pusch_data.tb_size;

    pusch_pdu->pusch_data.num_cb = 0; //CBG not supported
    //pusch_pdu->pusch_data.cb_present_and_position;
    //pusch_pdu->pusch_uci;
    //pusch_pdu->pusch_ptrs;
    //pusch_pdu->dfts_ofdm;
    //beamforming
    //pusch_pdu->beamforming; //not used for now


    ul_dci_request_pdu = &UL_dci_req->ul_dci_pdu_list[UL_dci_req->numPdus];
    memset((void*)ul_dci_request_pdu,0,sizeof(nfapi_nr_ul_dci_request_pdus_t));
    ul_dci_request_pdu->PDUType = NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE;
    ul_dci_request_pdu->PDUSize = (uint8_t)(2+sizeof(nfapi_nr_dl_tti_pdcch_pdu));
    nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu_rel15 = &ul_dci_request_pdu->pdcch_pdu.pdcch_pdu_rel15;
    UL_dci_req->numPdus+=1;


    LOG_I(MAC,"Configuring ULDCI/PDCCH in %d.%d\n", frameP,slotP);

    uint8_t nr_of_candidates, aggregation_level;
    find_aggregation_candidates(&aggregation_level, &nr_of_candidates, ss);
    NR_ControlResourceSet_t *coreset = get_coreset(bwp, ss, 1 /* dedicated */);
    const int cid = coreset->controlResourceSetId;
    const uint16_t Y = UE_info->Y[UE_id][cid][nr_mac->current_slot];
    const int m = UE_info->num_pdcch_cand[UE_id][cid];
    int CCEIndex = allocate_nr_CCEs(nr_mac,
                                    bwp,
                                    coreset,
                                    aggregation_level,
                                    Y,
                                    m,
                                    nr_of_candidates);
    if (CCEIndex < 0) {
      LOG_E(MAC, "%s(): CCE list not empty, couldn't schedule PUSCH\n", __func__);
      pusch_sched->active = false;
      return;
    }
    else {
      UE_info->num_pdcch_cand[UE_id][cid]++;
      nr_configure_pdcch(nr_mac,
                         pdcch_pdu_rel15,
                         UE_info->rnti[UE_id],
                         ss,
                         coreset,
                         scc,
                         bwp,
                         aggregation_level,
                         CCEIndex);

      dci_pdu_rel15_t *dci_pdu_rel15 = calloc(MAX_DCI_CORESET,sizeof(dci_pdu_rel15_t));
      config_uldci(ubwp,pusch_pdu,pdcch_pdu_rel15,&dci_pdu_rel15[0],dci_formats,rnti_types,time_domain_assignment,UE_info->UE_sched_ctrl[UE_id].tpc0,n_ubwp,bwp_id);
      fill_dci_pdu_rel15(scc,secondaryCellGroup,pdcch_pdu_rel15,dci_pdu_rel15,dci_formats,rnti_types,pusch_pdu->bwp_size,bwp_id);
      free(dci_pdu_rel15);
    }

#define LOG_PUSCH_PARAMES
#ifdef LOG_PUSCH_PARAMES
  static int islog =0;
  if(islog == 0)
  {   
      //islog = 1;
      LOG_I(MAC, "UE PUSCH PARAMS: frame %d, %d , rnti %d, bwpsize %d, bwpstart %d, subcarrier_spacing %d  cyclic_prefix %d, Nid_cell %d, num_slots_per_tdd %d\n",
      pusch_sched->frame,
      pusch_sched->slot,
      pusch_pdu->rnti,
      pusch_pdu->bwp_size,
      pusch_pdu->bwp_start,
      pusch_pdu->subcarrier_spacing,
      -1, //pusch_pdu->cyclic_prefix
      *scc->physCellId,
      num_slots_per_tdd
      );
      LOG_I(MAC, "UE PUSCH PARAMS: target_code_rate %d, qam_mod_order %d, mcs_index %d, mcs_table %d , transform_precoding %d, data_scrambling_id %d\n",
      pusch_pdu->target_code_rate,
      pusch_pdu->qam_mod_order,
      pusch_pdu->mcs_index,
      pusch_pdu->mcs_table,
      pusch_pdu->transform_precoding,
      pusch_pdu->data_scrambling_id);
      LOG_I(MAC, "UE PUSCH PARAMS: nrOfLayers %d, ul_dmrs_symb_pos %d dmrs_config_type %d, ul_dmrs_scrambling_id %d, scid %d, num_dmrs_cdm_grps_no_data %d\n",
      pusch_pdu->nrOfLayers,
      pusch_pdu->ul_dmrs_symb_pos,
      pusch_pdu->dmrs_config_type,
      pusch_pdu->ul_dmrs_scrambling_id,
      -1, //pusch_pdu->scid,
      pusch_pdu->num_dmrs_cdm_grps_no_data);
      LOG_I(MAC, "UE PUSCH PARAMS: dmrs_ports %d, resource_alloc %d, rb_start %d, rb_size %d, vrb_to_prb_mapping %d, frequency_hopping %d, tx_direct_current_location %d, uplink_frequency_shift_7p5khz %d, start_symbol_index %d, nr_of_symbols %d\n",
      pusch_pdu->dmrs_ports,
      pusch_pdu->resource_alloc,
      pusch_pdu->rb_start,
      pusch_pdu->rb_size,
      pusch_pdu->vrb_to_prb_mapping,
      -1, //pusch_pdu->frequency_hopping,
      -1, //pusch_pdu->tx_direct_current_location,
      pusch_pdu->uplink_frequency_shift_7p5khz,
       pusch_pdu->start_symbol_index,
      pusch_pdu->nr_of_symbols);
  }

#endif  
  }
}

