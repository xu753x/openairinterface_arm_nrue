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

/* \file proto.h
 * \brief MAC functions prototypes for gNB and UE
 * \author R. Knopp, K.H. HSU
 * \date 2018
 * \version 0.1
 * \company Eurecom / NTUST
 * \email: knopp@eurecom.fr, kai-hsiang.hsu@eurecom.fr
 * \note
 * \warning
 */

#ifndef __LAYER2_MAC_UE_PROTO_H__
#define __LAYER2_MAC_UE_PROTO_H__

#include "mac_defs.h"
#include "PHY/defs_nr_UE.h"
#include "RRC/NR_UE/rrc_defs.h"

/**\brief decode mib pdu in NR_UE, from if_module ul_ind with P7 tx_ind message
   \param module_id      module id
   \param cc_id          component carrier id
   \param gNB_index      gNB index
   \param extra_bits     extra bits for frame calculation
   \param l_ssb_equal_64 check if ssb number of candicate is equal 64, 1=equal; 0=non equal. Reference 38.212 7.1.1
   \param pduP           pointer to pdu
   \param pdu_length     length of pdu
   \param cell_id        cell id */
int8_t nr_ue_decode_mib(
    module_id_t module_id, 
    int cc_id, 
    uint8_t gNB_index, 
    uint8_t extra_bits, 
    uint32_t ssb_length, 
    uint32_t ssb_index,
    void *pduP, 
    uint16_t cell_id );

/**\brief decode SIB1 and other SIs pdus in NR_UE, from if_module dl_ind
   \param module_id      module id
   \param cc_id          component carrier id
   \param gNB_index      gNB index
   \param sibs_mask      sibs mask
   \param pduP           pointer to pdu
   \param pdu_length     length of pdu */
int8_t nr_ue_decode_BCCH_DL_SCH(module_id_t module_id,
                         int cc_id,
                         unsigned int gNB_index,
                         uint32_t sibs_mask,
                         uint8_t *pduP,
                         uint32_t pdu_len);

/**\brief primitive from RRC layer to MAC layer for configuration L1/L2, now supported 4 rrc messages: MIB, cell_group_config for MAC/PHY, spcell_config(serving cell config)
   \param module_id                 module id
   \param cc_id                     component carrier id
   \param gNB_index                 gNB index
   \param mibP                      pointer to RRC message MIB
   \param sccP                      pointer to ServingCellConfigCommon structure,
   \param spcell_configP            pointer to RRC message serving cell config*/
int nr_rrc_mac_config_req_ue(
    module_id_t                     module_id,
    int                             cc_idP,
    uint8_t                         gNB_index,
    NR_MIB_t                        *mibP,
    //NR_ServingCellConfigCommon_t    *sccP,
    NR_CellGroupConfig_t            *cell_group_config);

/**\brief initialization NR UE MAC instance(s), total number of MAC instance based on NB_NR_UE_MAC_INST*/
NR_UE_MAC_INST_t * nr_l2_init_ue(NR_UE_RRC_INST_t* rrc_inst);

/**\brief fetch MAC instance by module_id, within 0 - (NB_NR_UE_MAC_INST-1)
   \param module_id index of MAC instance(s)*/
NR_UE_MAC_INST_t *get_mac_inst(
    module_id_t module_id);

/**\brief called at each slot, slot length based on numerology. now use u=0, scs=15kHz, slot=1ms
          performs BSR/SR/PHR procedures, random access procedure handler and DLSCH/ULSCH procedures.
   \param dl_info     DL indication
   \param ul_info     UL indication*/
NR_UE_L2_STATE_t nr_ue_scheduler(nr_downlink_indication_t *dl_info, nr_uplink_indication_t *ul_info);

/**\brief fill nr_scheduled_response struct instance
   @param nr_scheduled_response_t *    pointer to scheduled_response instance to fill
   @param fapi_nr_dl_config_request_t* pointer to dl_config,
   @param fapi_nr_ul_config_request_t* pointer to ul_config,
   @param fapi_nr_tx_request_t*        pointer to tx_request;
   @param module_id_t mod_id           module ID
   @param int cc_id                    CC ID
   @param frame_t frame                frame number
   @param int slot                     reference number
   @param UE_nr_rxtx_proc_t *proc      pointer to process context */
void fill_scheduled_response(nr_scheduled_response_t *scheduled_response,
                             fapi_nr_dl_config_request_t *dl_config,
                             fapi_nr_ul_config_request_t *ul_config,
                             fapi_nr_tx_request_t *tx_request,
                             module_id_t mod_id,
                             int cc_id,
                             frame_t frame,
                             int slot,
                             int thread_id);

/* \brief Get SR payload (0,1) from UE MAC
@param Mod_id Instance id of UE in machine
@param CC_id Component Carrier index
@param eNB_id Index of eNB that UE is attached to
@param rnti C_RNTI of UE
@param subframe subframe number
@returns 0 for no SR, 1 for SR
*/
uint32_t ue_get_SR(module_id_t module_idP, int CC_id, frame_t frameP,
       uint8_t eNB_id, rnti_t rnti, sub_frame_t subframe);

int8_t nr_ue_get_SR(module_id_t module_idP, int CC_id, frame_t frameP, uint8_t eNB_id, uint16_t rnti, sub_frame_t subframe);

int8_t nr_ue_process_dci(module_id_t module_id, int cc_id, uint8_t gNB_index, frame_t frame, int slot, dci_pdu_rel15_t *dci, uint16_t rnti, uint8_t dci_format);
int nr_ue_process_dci_indication_pdu(module_id_t module_id, int cc_id, int gNB_index, frame_t frame, int slot, fapi_nr_dci_indication_pdu_t *dci);

uint32_t get_ssb_frame(uint32_t test);

uint32_t mr_ue_get_SR(module_id_t module_idP, int CC_id, frame_t frameP, uint8_t eNB_id, uint16_t rnti, sub_frame_t subframe);

/* \brief Get payload (MAC PDU) from UE PHY
@param dl_info            pointer to dl indication
@param ul_time_alignment  pointer to timing advance parameters
@param pdu_id             index of DL PDU
@returns void
*/
void nr_ue_send_sdu(nr_downlink_indication_t *dl_info,
                    NR_UL_TIME_ALIGNMENT_t *ul_time_alignment,
                    int pdu_id);

void nr_ue_process_mac_pdu(nr_downlink_indication_t *dl_info,
                           NR_UL_TIME_ALIGNMENT_t *ul_time_alignment,
                           int pdu_id);

uint16_t nr_generate_ulsch_pdu(uint8_t *sdus_payload,
                                    uint8_t *pdu,
                                    uint8_t num_sdus,
                                    uint16_t *sdu_lengths,
                                    uint8_t *sdu_lcids,
                                    uint8_t power_headroom,
                                    uint16_t crnti,
                                    uint16_t truncated_bsr,
                                    uint16_t short_bsr,
                                    uint16_t long_bsr,
                                    unsigned short post_padding,
                                    uint16_t buflen);

void ue_dci_configuration(NR_UE_MAC_INST_t *mac, fapi_nr_dl_config_request_t *dl_config, frame_t frame, int slot);

uint8_t nr_extract_dci_info(NR_UE_MAC_INST_t *mac,
                            uint8_t dci_format,
                            uint8_t dci_length,
                            uint16_t rnti,
                            uint64_t *dci_pdu,
                            dci_pdu_rel15_t *nr_pdci_info_extracted);

int8_t nr_ue_process_dci_time_dom_resource_assignment(NR_UE_MAC_INST_t *mac,
                                                      nfapi_nr_ue_pusch_pdu_t *pusch_config_pdu,
                                                      fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config_pdu,
                                                      uint8_t time_domain_ind,
                                                      bool use_default);

uint8_t
nr_ue_get_sdu(module_id_t module_idP, int CC_id, frame_t frameP,
           sub_frame_t subframe, uint8_t eNB_index,
           uint8_t *ulsch_buffer, uint16_t buflen, uint8_t *access_mode) ;

int set_tdd_config_nr_ue(fapi_nr_config_request_t *cfg, int mu,
                         int nrofDownlinkSlots, int nrofDownlinkSymbols,
                         int nrofUplinkSlots,   int nrofUplinkSymbols);

/** \brief Function for UE/PHY to compute PUSCH transmit power in power-control procedure.
    @param Mod_id Module id of UE
    @returns Po_NOMINAL_PUSCH (PREAMBLE_RECEIVED_TARGET_POWER+DELTA_PREAMBLE
*/
int nr_get_Po_NOMINAL_PUSCH(NR_PRACH_RESOURCES_t *prach_resources, module_id_t module_idP, uint8_t CC_id);

/** \brief Function to compute DELTA_PREAMBLE from 38.321 subclause 7.3
   (for RA power ramping procedure and Msg3 PUSCH power control policy)
    @param Mod_id Module id of UE
    @returns DELTA_PREAMBLE
*/
int8_t nr_get_DELTA_PREAMBLE(module_id_t mod_id, int CC_id, uint16_t prach_format);

/** \brief Function to compute configured maximum output power according to clause 6.2.4 of 3GPP TS 38.101-1 version 16.5.0 Release 16
    @param Mod_id Module id of UE
*/
long nr_get_Pcmax(module_id_t mod_id);

/* Random Access */

/* \brief This function schedules the PRACH according to prach_ConfigurationIndex and TS 38.211 tables 6.3.3.2.x
and fills the PRACH PDU per each FD occasion.
@param module_idP Index of UE instance
@param frameP Frame index
@param slotP Slot index
@param thread_id RX/TX Thread ID
@returns void
*/
void nr_ue_prach_scheduler(module_id_t module_idP, frame_t frameP, sub_frame_t slotP, int thread_id);

/* \brief This function schedules the Msg3 transmission
@param
@param
@param
@returns void
*/
void nr_ue_msg3_scheduler(NR_UE_MAC_INST_t *mac,
                          frame_t current_frame,
                          sub_frame_t current_slot,
                          uint8_t Msg3_tda_id);

/* \brief Function called by PHY to process the received RAR and check that the preamble matches what was sent by the gNB. It provides the timing advance and t-CRNTI.
@param Mod_id Index of UE instance
@param CC_id Index to a component carrier
@param frame Frame index
@param ra_rnti RA_RNTI value
@param dlsch_buffer  Pointer to dlsch_buffer containing RAR PDU
@param t_crnti Pointer to PHY variable containing the T_CRNTI
@param preamble_index Preamble Index used by PHY to transmit the PRACH.  This should match the received RAR to trigger the rest of
random-access procedure
@param selected_rar_buffer the output buffer for storing the selected RAR header and RAR payload
@returns timing advance or 0xffff if preamble doesn't match
*/
int nr_ue_process_rar(nr_downlink_indication_t *dl_info, NR_UL_TIME_ALIGNMENT_t *ul_time_alignment, int pdu_id);

void nr_process_rar(nr_downlink_indication_t *dl_info);

void nr_ue_contention_resolution(module_id_t module_id, int cc_id, frame_t frame, int slot, NR_PRACH_RESOURCES_t *prach_resources);

void nr_ra_failed(uint8_t mod_id, uint8_t CC_id, NR_PRACH_RESOURCES_t *prach_resources, frame_t frame, int slot);

void nr_ra_succeeded(module_id_t mod_id, frame_t frame, int slot);

/* \brief Function called by PHY to retrieve information to be transmitted using the RA procedure.
If the UE is not in PUSCH mode for a particular eNB index, this is assumed to be an Msg3 and MAC
attempts to retrieves the CCCH message from RRC. If the UE is in PUSCH mode for a particular eNB
index and PUCCH format 0 (Scheduling Request) is not activated, the MAC may use this resource for
andom-access to transmit a BSR along with the C-RNTI control element (see 5.1.4 from 36.321)
@param mod_id Index of UE instance
@param CC_id Component Carrier Index
@param frame
@param gNB_id gNB index
@param nr_slot_tx slot for PRACH transmission
@returns indication to generate PRACH to phy */
uint8_t nr_ue_get_rach(NR_PRACH_RESOURCES_t *prach_resources,
                       fapi_nr_ul_config_prach_pdu *prach_pdu,
                       module_id_t mod_id,
                       int CC_id,
                       frame_t frame,
                       uint8_t gNB_id,
                       int nr_slot_tx);

/* \brief Function implementing the routine for the selection of Random Access resources (5.1.2 TS 38.321).
@param module_idP Index of UE instance
@param CC_id Component Carrier Index
@param gNB_index gNB index
@param rach_ConfigDedicated
@returns void */
void nr_get_prach_resources(module_id_t mod_id,
                            int CC_id,
                            uint8_t gNB_id,
                            NR_PRACH_RESOURCES_t *prach_resources,
                            fapi_nr_ul_config_prach_pdu *prach_pdu,
                            NR_RACH_ConfigDedicated_t * rach_ConfigDedicated);

void nr_Msg1_transmitted(module_id_t mod_id, uint8_t CC_id, frame_t frameP, uint8_t gNB_id);

void nr_Msg3_transmitted(module_id_t mod_id, uint8_t CC_id, frame_t frameP, uint8_t gNB_id);

void nr_ue_msg2_scheduler(module_id_t mod_id, uint16_t rach_frame, uint16_t rach_slot, uint16_t *msg2_frame, uint16_t *msg2_slot);

int8_t nr_ue_process_dci_freq_dom_resource_assignment(nfapi_nr_ue_pusch_pdu_t *pusch_config_pdu,
                                                      fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config_pdu,
                                                      uint16_t n_RB_ULBWP,
                                                      uint16_t n_RB_DLBWP,
                                                      uint16_t riv);

void get_num_re_dmrs(nfapi_nr_ue_pusch_pdu_t *pusch_pdu,
                     uint8_t *nb_dmrs_re_per_rb,
                     uint16_t *number_dmrs_symbols);

void build_ssb_to_ro_map(NR_ServingCellConfigCommon_t *scc, uint8_t unpaired);

void config_bwp_ue(NR_UE_MAC_INST_t *mac, uint16_t *bwp_ind, uint8_t *dci_format);

fapi_nr_ul_config_request_t *get_ul_config_request(NR_UE_MAC_INST_t *mac, int slot);

void fill_ul_config(fapi_nr_ul_config_request_t *ul_config, frame_t frame_tx, int slot_tx, uint8_t pdu_type);

// PUSCH scheduler:
// - Calculate the slot in which ULSCH should be scheduled. This is current slot + K2,
// - where K2 is the offset between the slot in which UL DCI is received and the slot
// - in which ULSCH should be scheduled. K2 is configured in RRC configuration.  
// PUSCH Msg3 scheduler:
// - scheduled by RAR UL grant according to 8.3 of TS 38.213
int nr_ue_pusch_scheduler(NR_UE_MAC_INST_t *mac, uint8_t is_Msg3, frame_t current_frame, int current_slot, frame_t *frame_tx, int *slot_tx, uint8_t tda_id);

int get_rnti_type(NR_UE_MAC_INST_t *mac, uint16_t rnti);

// Configuration of Msg3 PDU according to clauses:
// - 8.3 of 3GPP TS 38.213 version 16.3.0 Release 16
// - 6.1.2.2 of TS 38.214
// - 6.1.3 of TS 38.214
// - 6.2.2 of TS 38.214
// - 6.1.4.2 of TS 38.214
// - 6.4.1.1.1 of TS 38.211
// - 6.3.1.7 of 38.211
int nr_config_pusch_pdu(NR_UE_MAC_INST_t *mac,
                        nfapi_nr_ue_pusch_pdu_t *pusch_config_pdu,
                        dci_pdu_rel15_t *dci,
                        RAR_grant_t *rar_grant,
                        uint16_t rnti,
                        uint8_t *dci_format);
#endif
/** @}*/
