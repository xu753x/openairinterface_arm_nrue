


/* Author R. Knopp / EURECOM / OpenAirInterface.org */
#ifndef __NR_REFSIG_DEFS__H__
#define __NR_REFSIG_DEFS__H__

#include "PHY/defs_nr_UE.h"
#include "PHY/LTE_REFSIG/lte_refsig.h"


/*!\brief This function generates the NR Gold sequence (38-211, Sec 5.2.1) for the PBCH DMRS.
@param PHY_VARS_NR_UE* ue structure provides configuration, frame parameters and the pointers to the 32 bits sequence storage tables
 */
int nr_pbch_dmrs_rx(int dmrss,
                    unsigned int *nr_gold_pbch,
                    int32_t *output);

/*!\brief This function generates the NR Gold sequence (38-211, Sec 5.2.1) for the PDCCH DMRS.
@param PHY_VARS_NR_UE* ue structure provides configuration, frame parameters and the pointers to the 32 bits sequence storage tables
 */
int nr_pdcch_dmrs_rx(PHY_VARS_NR_UE *ue,
                     uint8_t eNB_offset,
                     unsigned int Ns,
                     unsigned int *nr_gold_pdcch,
                     int32_t *output,
                     unsigned short p,
                     unsigned short nb_rb_corset);

int nr_pdsch_dmrs_rx(PHY_VARS_NR_UE *ue,
                     unsigned int Ns,
                     unsigned int *nr_gold_pdsch,
                     int32_t *output,
                     unsigned short p,
                     unsigned char lp,
                     unsigned short nb_pdsch_rb);

void nr_gold_pbch(PHY_VARS_NR_UE* ue);

void nr_gold_pdcch(PHY_VARS_NR_UE* ue,
                   unsigned short n_idDMRS);

void nr_gold_pdsch(PHY_VARS_NR_UE* ue,
                   unsigned short *n_idDMRS);

void nr_init_pusch_dmrs(PHY_VARS_NR_UE* ue,
                        uint16_t *N_n_scid,
                        uint8_t n_scid);

#endif
