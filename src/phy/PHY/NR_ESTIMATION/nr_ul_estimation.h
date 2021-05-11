

#ifndef __NR_UL_ESTIMATION_DEFS__H__
#define __NR_UL_ESTIMATION_DEFS__H__


#include "PHY/defs_gNB.h"



/*!
\brief This function performs channel estimation including frequency interpolation
\param gNB Pointer to gNB PHY variables
\param Ns slot number (0..19)
\param p
\param symbol symbol within slot
\param bwp_start_subcarrier, first allocated subcarrier
\param nb_rb_pusch, number of allocated RBs for this UE
*/

int nr_pusch_channel_estimation(PHY_VARS_gNB *gNB,
                                unsigned char Ns,
                                unsigned short p,
                                unsigned char symbol,
                                int ul_id,
                                unsigned short bwp_start_subcarrier,
                                nfapi_nr_pusch_pdu_t *pusch_pdu);

void dump_nr_I0_stats(FILE *fd,PHY_VARS_gNB *gNB);

void gNB_I0_measurements(PHY_VARS_gNB *gNB,int first_symb,int num_symb);

void nr_gnb_measurements(PHY_VARS_gNB *gNB, uint8_t ulsch_id, unsigned char harq_pid, unsigned char symbol);

int nr_est_timing_advance_pusch(PHY_VARS_gNB* phy_vars_gNB, int UE_id);

void nr_pusch_ptrs_processing(PHY_VARS_gNB *gNB,
                              NR_DL_FRAME_PARMS *frame_parms,
                              nfapi_nr_pusch_pdu_t *rel15_ul,
                              uint8_t ulsch_id,
                              uint8_t nr_tti_rx,
                              unsigned char symbol,
                              uint32_t nb_re_pusch);
#endif
