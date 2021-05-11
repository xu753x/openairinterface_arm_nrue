



#ifndef __NR_DLSCH__H
#define __NR_DLSCH__H

#include "PHY/defs_gNB.h"

void nr_get_time_domain_allocation_type(nfapi_nr_config_request_t config,
                                        nfapi_nr_dl_tti_pdcch_pdu dci_pdu,
                                        nfapi_nr_dl_tti_pdsch_pdu *pdsch_pdu);

void nr_check_time_alloc(uint8_t S, uint8_t L,nfapi_nr_dl_tti_pdsch_pdu_rel15_t *rel15,nfapi_nr_config_request_t *cfg);

uint16_t get_RIV(uint16_t rb_start, uint16_t L, uint16_t N_RB);

uint16_t get_SLIV(uint8_t S, uint8_t L);

uint8_t nr_get_S(uint8_t row_idx, uint8_t CP, uint8_t time_alloc_type, uint8_t dmrs_typeA_position);

void nr_get_rbg_parms(NR_BWP_PARMS* bwp, uint8_t config_type);

void nr_get_rbg_list(uint32_t bitmap, uint8_t n_rbg, uint8_t* rbg_list);


uint8_t nr_get_Qm(uint8_t Imcs, uint8_t table_idx);

uint32_t nr_get_code_rate(uint8_t Imcs, uint8_t table_idx);

void nr_pdsch_codeword_scrambling(uint8_t *in,
                                  uint32_t size,
                                  uint8_t q,
                                  uint32_t Nid,
                                  uint32_t n_RNTI,
                                  uint32_t* out);

void nr_fill_dlsch(PHY_VARS_gNB *gNB,
                   int frame,
                   int slot,
                   nfapi_nr_dl_tti_pdsch_pdu *pdsch_pdu,
                   unsigned char *sdu); 

uint8_t nr_generate_pdsch(PHY_VARS_gNB *gNB,
			  int frame,
			  int slot);
void free_gNB_dlsch(NR_gNB_DLSCH_t **dlschptr, uint16_t N_RB);

void clean_gNB_dlsch(NR_gNB_DLSCH_t *dlsch);

void clean_gNB_ulsch(NR_gNB_ULSCH_t *ulsch);

int16_t find_nr_dlsch(uint16_t rnti, PHY_VARS_gNB *gNB,find_type_t type);

NR_gNB_SCH_STATS_t *find_nr_dlsch_stats(uint16_t rnti, PHY_VARS_gNB *gNB,find_type_t type);

int nr_dlsch_encoding(PHY_VARS_gNB *gNB,
		      unsigned char *a,int frame,
		      uint8_t slot,
		      NR_gNB_DLSCH_t *dlsch,
		      NR_DL_FRAME_PARMS* frame_parms,
		      time_stats_t *tinput,
		      time_stats_t *tprep,
		      time_stats_t *tparity,
		      time_stats_t *toutput,
		      time_stats_t *dlsch_rate_matching_stats,
		      time_stats_t *dlsch_interleaving_stats,
		      time_stats_t *dlsch_segmentation_stats);

void init_dlsch_tpool(uint8_t nun_dlsch_threads);
void nr_emulate_dlsch_payload(uint8_t* payload, uint16_t size);

void dump_pdsch_stats(PHY_VARS_gNB *gNB);

void clear_pdsch_stats(PHY_VARS_gNB *gNB);

#endif
