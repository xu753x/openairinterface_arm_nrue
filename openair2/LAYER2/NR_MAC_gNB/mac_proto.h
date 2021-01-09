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

/*! \file mac_proto.h
 * \brief MAC functions prototypes for gNB
 * \author Navid Nikaein and Raymond Knopp, WEI-TAI CHEN
 * \date 2010 - 2014, 2018
 * \email navid.nikaein@eurecom.fr, kroempa@gmail.com
 * \version 1.0
 * \company Eurecom, NTUST
 */

#ifndef __LAYER2_NR_MAC_PROTO_H__
#define __LAYER2_NR_MAC_PROTO_H__

#include "PHY/defs_gNB.h"

#include "LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "NR_TAG-Id.h"

void set_cset_offset(uint16_t);

void mac_top_init_gNB(void);

void config_common(int Mod_idP,
                   int pdsch_AntennaPorts,
		   NR_ServingCellConfigCommon_t *scc
		   );

int rrc_mac_config_req_gNB(module_id_t Mod_idP, 
			   int ssb_SubcarrierOffset,
                           int pdsch_AntennaPorts,
                           int pusch_tgt_snrx10,
                           int pucch_tgt_snrx10,
                           NR_ServingCellConfigCommon_t *scc,
			   int nsa_flag,
			   uint32_t rnti,
			   NR_CellGroupConfig_t *secondaryCellGroup
                           );

void clear_nr_nfapi_information(gNB_MAC_INST * gNB, 
                                int CC_idP,
                                frame_t frameP, 
                                sub_frame_t subframeP);

void gNB_dlsch_ulsch_scheduler(module_id_t module_idP,
			       frame_t frame_rxP, sub_frame_t slot_rxP);

int nr_generate_dlsch_pdu(module_id_t Mod_idP,
                          NR_UE_sched_ctrl_t *ue_sched_ctl,
                          unsigned char *sdus_payload,
                          unsigned char *mac_pdu,
                          unsigned char num_sdus,
                          unsigned short *sdu_lengths,
                          unsigned char *sdu_lcids,
                          unsigned char drx_cmd,
                          unsigned char *ue_cont_res_id,
                          unsigned short post_padding);

void nr_schedule_ue_spec(module_id_t module_id,
                         frame_t frame,
                         sub_frame_t slot,
                         int num_slots_per_tdd);

/* \brief default preprocessor */
void nr_simple_dlsch_preprocessor(module_id_t module_id,
                                  frame_t frame,
                                  sub_frame_t slot,
                                  int num_slots_per_tdd);

void schedule_nr_mib(module_id_t module_idP, frame_t frameP, sub_frame_t subframeP, uint8_t slots_per_frame);

/// uplink scheduler
void nr_schedule_ulsch(module_id_t module_id,
                       frame_t frame,
                       sub_frame_t slot,
                       int num_slots_per_tdd,
                       int ul_slots,
                       uint64_t ulsch_in_slot_bitmap);

void nr_simple_ulsch_preprocessor(module_id_t module_id,
                                  frame_t frame,
                                  sub_frame_t slot,
                                  int num_slots_per_tdd,
                                  uint64_t ulsch_in_slot_bitmap);

/////// Random Access MAC-PHY interface functions and primitives ///////

void nr_schedule_RA(module_id_t module_idP, frame_t frameP, sub_frame_t slotP);

/* \brief Function to indicate a received preamble on PRACH.  It initiates the RA procedure.
@param module_idP Instance ID of gNB
@param preamble_index index of the received RA request
@param slotP Slot number on which to act
@param timing_offset Offset in samples of the received PRACH w.r.t. eNB timing. This is used to
@param rnti RA rnti corresponding to this PRACH preamble
@param rach_resource type (0=non BL/CE,1 CE level 0,2 CE level 1, 3 CE level 2,4 CE level 3)
*/
void nr_initiate_ra_proc(module_id_t module_idP, int CC_id, frame_t frameP, sub_frame_t slotP,
                         uint16_t preamble_index, uint8_t freq_index, uint8_t symbol, int16_t timing_offset);

void nr_clear_ra_proc(module_id_t module_idP, int CC_id, frame_t frameP);

int nr_allocate_CCEs(int module_idP, int CC_idP, frame_t frameP, sub_frame_t slotP, int test_only);

void nr_get_Msg3alloc(module_id_t module_id,
                      int CC_id,
                      NR_ServingCellConfigCommon_t *scc,
                      NR_BWP_Uplink_t *ubwp,
                      sub_frame_t current_subframe,
                      frame_t current_frame,
                      NR_RA_t *ra);

/* \brief Function in gNB to fill RAR pdu when requested by PHY.
@param ra Instance of RA resources of gNB
@param dlsch_buffer Pointer to RAR input buffer
@param N_RB_UL Number of UL resource blocks
*/
void nr_fill_rar(uint8_t Mod_idP,
                 NR_RA_t * ra,
                 uint8_t * dlsch_buffer,
                 nfapi_nr_pusch_pdu_t  *pusch_pdu);


void schedule_nr_prach(module_id_t module_idP, frame_t frameP, sub_frame_t slotP);

uint16_t nr_mac_compute_RIV(uint16_t N_RB_DL, uint16_t RBstart, uint16_t Lcrbs);

/////// Phy test scheduler ///////

/* \brief preprocessor for phytest: schedules UE_id 0 with fixed MCS on all
 * freq resources */
void nr_preprocessor_phytest(module_id_t module_id,
                             frame_t frame,
                             sub_frame_t slot,
                             int num_slots_per_tdd);
/* \brief UL preprocessor for phytest: schedules UE_id 0 with fixed MCS on a
 * fixed set of resources */
void nr_ul_preprocessor_phytest(module_id_t module_id,
                                frame_t frame,
                                sub_frame_t slot,
                                int num_slots_per_tdd,
                                uint64_t ulsch_in_slot_bitmap);

void nr_schedule_css_dlsch_phytest(module_id_t   module_idP,
                                   frame_t       frameP,
                                   sub_frame_t   subframeP);

void nr_fill_nfapi_dl_pdu(int Mod_id,
                          nfapi_nr_dl_tti_request_body_t *dl_req,
                          rnti_t rnti,
                          NR_CellGroupConfig_t *secondaryCellGroup,
                          NR_UE_sched_ctrl_t *sched_ctrl,
                          NR_sched_pucch_t *pucch_sched,
                          nfapi_nr_dmrs_type_e dmrsConfigType,
                          uint16_t R,
                          uint8_t Qm,
                          uint32_t tbs,
                          int StartSymbolIndex,
                          int NrOfSymbols,
                          int harq_pid,
                          int ndi,
                          int round);

void handle_nr_uci_pucch_0_1(module_id_t mod_id,
                             frame_t frame,
                             sub_frame_t slot,
                             const nfapi_nr_uci_pucch_pdu_format_0_1_t *uci_01);
void handle_nr_uci_pucch_2_3_4(module_id_t mod_id,
                               frame_t frame,
                               sub_frame_t slot,
                               const nfapi_nr_uci_pucch_pdu_format_2_3_4_t *uci_234);


void config_uldci(NR_BWP_Uplink_t *ubwp,
                  nfapi_nr_pusch_pdu_t *pusch_pdu,
                  nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu_rel15,
                  dci_pdu_rel15_t *dci_pdu_rel15,
                  int *dci_formats,
                  int time_domain_assignment, uint8_t tpc,
                  int n_ubwp, int bwp_id);

void nr_schedule_pucch(int Mod_idP,
                       int UE_id,
                       int nr_ulmix_slots,
                       frame_t frameP,
                       sub_frame_t slotP);

void csi_period_offset(NR_CSI_ReportConfig_t *csirep,
                       int *period, int *offset);

void nr_csi_meas_reporting(int Mod_idP,
                           int UE_id,
                           frame_t frameP,
                           sub_frame_t slotP,
                           int slots_per_tdd,
                           int ul_slots,
                           int n_slots_frame);

bool nr_acknack_scheduling(int Mod_idP,
                           int UE_id,
                           frame_t frameP,
                           sub_frame_t slotP);

void get_pdsch_to_harq_feedback(int Mod_idP,
                                int UE_id,
                                NR_SearchSpace__searchSpaceType_PR ss_type,
                                uint8_t *pdsch_to_harq_feedback);
  
void nr_configure_css_dci_initial(nfapi_nr_dl_tti_pdcch_pdu_rel15_t* pdcch_pdu,
                                  nr_scs_e scs_common,
                                  nr_scs_e pdcch_scs,
                                  nr_frequency_range_e freq_range,
                                  uint8_t rmsi_pdcch_config,
                                  uint8_t ssb_idx,
                                  uint8_t k_ssb,
                                  uint16_t sfn_ssb,
                                  uint8_t n_ssb,
                                  uint16_t nb_slots_per_frame,
                                  uint16_t N_RB);
/*
int nr_is_dci_opportunity(nfapi_nr_search_space_t search_space,
                          nfapi_nr_coreset_t coreset,
                          uint16_t frame,
                          uint16_t slot,
                          nfapi_nr_config_request_scf_t cfg);
*/

void nr_configure_pucch(nfapi_nr_pucch_pdu_t* pucch_pdu,
			NR_ServingCellConfigCommon_t *scc,
			NR_BWP_Uplink_t *bwp,
                        uint16_t rnti,
                        uint8_t pucch_resource,
                        uint16_t O_csi,
                        uint16_t O_ack,
                        uint8_t O_sr);

void find_search_space(int ss_type,
                       NR_BWP_Downlink_t *bwp,
                       NR_SearchSpace_t *ss);

void nr_configure_pdcch(gNB_MAC_INST *nr_mac,
                        nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu,
                        uint16_t rnti,
                        NR_SearchSpace_t *ss,
                        NR_ControlResourceSet_t *coreset,
                        NR_ServingCellConfigCommon_t *scc,
                        NR_BWP_Downlink_t *bwp,
                        uint8_t aggregation_level,
                        int CCEIndex);

void fill_dci_pdu_rel15(NR_ServingCellConfigCommon_t *scc,
                        NR_CellGroupConfig_t *secondaryCellGroup,
                        nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu_rel15,
                        dci_pdu_rel15_t *dci_pdu_rel15,
                        int *dci_formats,
                        int *rnti_types,
                        int N_RB,
                        int bwp_id);

void prepare_dci(NR_CellGroupConfig_t *secondaryCellGroup,
                 dci_pdu_rel15_t *dci_pdu_rel15,
                 nr_dci_format_t format,
                 int bwp_id);

/* find coreset within the search space */
NR_ControlResourceSet_t *get_coreset(NR_BWP_Downlink_t *bwp,
                                     NR_SearchSpace_t *ss,
                                     int ss_type);

/* find a search space within a BWP */
NR_SearchSpace_t *get_searchspace(
    NR_BWP_Downlink_t *bwp,
    NR_SearchSpace__searchSpaceType_PR target_ss);

void find_aggregation_candidates(uint8_t *aggregation_level,
                                 uint8_t *nr_of_candidates,
                                 NR_SearchSpace_t *ss);

long get_K2(NR_BWP_Uplink_t *ubwp, int time_domain_assignment, int mu);

void nr_save_pusch_fields(const NR_ServingCellConfigCommon_t *scc,
                          const NR_BWP_Uplink_t *ubwp,
                          long dci_format,
                          int tda,
                          uint8_t num_dmrs_cdm_grps_no_data,
                          NR_sched_pusch_save_t *ps);

uint8_t nr_get_tpc(int target, uint8_t cqi, int incr);

int get_spf(nfapi_nr_config_request_scf_t *cfg);

int to_absslot(nfapi_nr_config_request_scf_t *cfg,int frame,int slot);

void nr_get_tbs_dl(nfapi_nr_dl_tti_pdsch_pdu *pdsch_pdu,
		   int x_overhead,
                   uint8_t numdmrscdmgroupnodata,
                   uint8_t tb_scaling);
/** \brief Computes Q based on I_MCS PDSCH and table_idx for downlink. Implements MCS Tables from 38.214. */
uint8_t nr_get_Qm_dl(uint8_t Imcs, uint8_t table_idx);
uint32_t nr_get_code_rate_dl(uint8_t Imcs, uint8_t table_idx);

/** \brief Computes Q based on I_MCS PDSCH and table_idx for uplink. Implements MCS Tables from 38.214. */
uint8_t nr_get_Qm_ul(uint8_t Imcs, uint8_t table_idx);
uint32_t nr_get_code_rate_ul(uint8_t Imcs, uint8_t table_idx);

int NRRIV2BW(int locationAndBandwidth,int N_RB);

int NRRIV2PRBOFFSET(int locationAndBandwidth,int N_RB);

void dump_nr_ue_list(NR_UE_list_t *listP);
void add_nr_ue_list(NR_UE_list_t *listP, int UE_id);

int find_nr_UE_id(module_id_t mod_idP, rnti_t rntiP);

int find_nr_RA_id(module_id_t mod_idP, int CC_idP, rnti_t rntiP);

int add_new_nr_ue(module_id_t mod_idP, rnti_t rntiP);

void mac_remove_nr_ue(module_id_t mod_id, rnti_t rnti);

int allocate_nr_CCEs(gNB_MAC_INST *nr_mac,
                     NR_BWP_Downlink_t *bwp,
                     NR_ControlResourceSet_t *coreset,
                     int aggregation,
                     uint16_t Y,
                     int m,
                     int nr_of_candidates);

uint16_t compute_pucch_prb_size(uint8_t format,
                                uint8_t nr_prbs,
                                uint16_t O_tot,
                                uint16_t O_csi,
                                NR_PUCCH_MaxCodeRate_t *maxCodeRate,
                                uint8_t Qm,
                                uint8_t n_symb,
                                uint8_t n_re_ctrl);

void compute_csi_bitlen (NR_CellGroupConfig_t *secondaryCellGroup, NR_UE_info_t *UE_info, int UE_id);

int get_dlscs(nfapi_nr_config_request_t *cfg);

int get_ulscs(nfapi_nr_config_request_t *cfg);

int get_symbolsperslot(nfapi_nr_config_request_t *cfg);

void config_nr_mib(int Mod_idP, 
                   int CC_idP,
                   int p_gNBP,
                   int subCarrierSpacingCommon,
                   uint32_t ssb_SubcarrierOffset,
                   int dmrs_TypeA_Position,
                   uint32_t pdcch_ConfigSIB1,
                   int cellBarred,
                   int intraFreqReselection);


void nr_generate_Msg2(module_id_t module_idP,
                      int CC_id,
                      frame_t frameP,
                      sub_frame_t slotP);

void nr_process_mac_pdu(
    module_id_t module_idP,
    rnti_t rnti,
    uint8_t CC_id,
    frame_t frameP,
    uint8_t *pduP,
    uint16_t mac_pdu_len);

int binomial(int n, int k);

bool is_xlsch_in_slot(uint64_t bitmap, sub_frame_t slot);

void fill_ssb_vrb_map (NR_COMMON_channels_t *cc, int rbStart, int CC_id);


/* \brief Function to indicate a received SDU on ULSCH.
@param Mod_id Instance ID of gNB
@param CC_id Component carrier index
@param rnti RNTI of UE transmitting the SDU
@param sdu Pointer to received SDU
@param sdu_len Length of SDU
@param timing_advance timing advance adjustment after this pdu
@param ul_cqi Uplink CQI estimate after this pdu (SNR quantized to 8 bits, -64 ... 63.5 dB in .5dB steps)
*/
void nr_rx_sdu(const module_id_t gnb_mod_idP,
               const int CC_idP,
               const frame_t frameP,
               const sub_frame_t subframeP,
               const rnti_t rntiP,
               uint8_t * sduP,
               const uint16_t sdu_lenP,
               const uint16_t timing_advance,
               const uint8_t ul_cqi,
               const uint16_t rssi);

void handle_nr_ul_harq(module_id_t mod_id,
                       frame_t frame,
                       sub_frame_t slot,
                       const nfapi_nr_crc_t *crc_pdu);

int16_t ssb_index_from_prach(module_id_t module_idP,
                             frame_t frameP,
			     sub_frame_t slotP,
                             uint16_t preamble_index,
                             uint8_t freq_index,
                             uint8_t symbol);

void find_SSB_and_RO_available(module_id_t module_idP);

#endif /*__LAYER2_NR_MAC_PROTO_H__*/
