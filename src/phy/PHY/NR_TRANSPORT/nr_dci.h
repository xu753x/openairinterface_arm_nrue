

#ifndef __PHY_NR_TRANSPORT_DCI__H
#define __PHY_NR_TRANSPORT_DCI__H

#include "PHY/defs_gNB.h"
#include "PHY/NR_REFSIG/nr_refsig.h"

uint16_t nr_get_dci_size(nfapi_nr_dci_format_e format,
                         nfapi_nr_rnti_type_e rnti_type,
                         uint16_t N_RB);

void nr_generate_dci_top(PHY_VARS_gNB *gNB,
			    nfapi_nr_dl_tti_pdcch_pdu *pdcch_pdu,
			    nfapi_nr_dl_tti_pdcch_pdu *ul_pdcch_pdu,
			    uint32_t **gold_pdcch_dmrs,
                            int32_t *txdataF,
                            int16_t amp,
                            NR_DL_FRAME_PARMS frame_parms);

void nr_pdcch_scrambling(uint32_t *in,
                         uint32_t size,
                         uint32_t Nid,
                         uint32_t n_RNTI,
                         uint32_t *out);

int16_t find_nr_pdcch(int frame,int slot, PHY_VARS_gNB *gNB,find_type_t type);

void nr_fill_dci(PHY_VARS_gNB *gNB,
                 int frame,
                 int slot,
		 nfapi_nr_dl_tti_pdcch_pdu *pdcch_pdu);

int16_t find_nr_ul_dci(int frame,int slot, PHY_VARS_gNB *gNB,find_type_t type);

void nr_fill_ul_dci(PHY_VARS_gNB *gNB,
		    int frame,
		    int slot,
		    nfapi_nr_ul_dci_request_pdus_t *pdcch_pdu);

void nr_fill_cce_list(PHY_VARS_gNB *gNB, uint8_t m,nfapi_nr_dl_tti_pdcch_pdu_rel15_t *);

void get_coreset_rballoc(uint8_t *FreqDomainResource,int *n_rb,int *rb_offset);

#endif //__PHY_NR_TRANSPORT_DCI__H
