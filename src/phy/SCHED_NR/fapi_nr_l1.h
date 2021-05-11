



#include "PHY/defs_gNB.h"
#include "PHY/phy_extern.h"
#include "SCHED_NR/sched_nr.h"
#include "nfapi_nr_interface.h"
#include "nfapi_nr_interface_scf.h"

// added
void handle_nr_nfapi_ssb_pdu(PHY_VARS_gNB *gNB,
						int frame,int slot,
						nfapi_nr_dl_tti_request_pdu_t *dl_tti_pdu);

void nr_schedule_response(NR_Sched_Rsp_t *Sched_INFO);

void handle_nfapi_nr_pdcch_pdu(PHY_VARS_gNB *gNB,
			       int frame, int subframe,
			       nfapi_nr_dl_tti_pdcch_pdu *dcl_dl_pdu);

void handle_nr_nfapi_pdsch_pdu(PHY_VARS_gNB *gNB,int frame,int slot,
			       nfapi_nr_dl_tti_pdsch_pdu *pdsch_pdu,
                            uint8_t *sdu);


void nr_fill_indication(PHY_VARS_gNB *gNB, int frame, int slot_rx, int UE_id, uint8_t harq_pid, uint8_t crc_flag);
//added

void handle_nfapi_nr_ul_dci_pdu(PHY_VARS_gNB *gNB,
			       int frame, int slot,
			       nfapi_nr_ul_dci_request_pdus_t *ul_dci_request_pdu);
