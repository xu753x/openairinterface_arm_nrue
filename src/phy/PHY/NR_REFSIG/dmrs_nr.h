



#ifndef DMRS_NR_H
#define DMRS_NR_H

#include "PHY/defs_nr_UE.h"
#include "PHY/types.h"
#include "PHY/NR_REFSIG/ss_pbch_nr.h"
#include "PHY/NR_REFSIG/pss_nr.h"
#include "PHY/NR_REFSIG/sss_nr.h"

/************** CODE GENERATION ***********************************/

/************** DEFINE ********************************************/


/************* STRUCTURES *****************************************/


/************** VARIABLES *****************************************/

/************** FUNCTION ******************************************/

int pseudo_random_sequence(int M_PN, uint32_t *c, uint32_t cinit);
void lte_gold_new(LTE_DL_FRAME_PARMS *frame_parms, uint32_t lte_gold_table[20][2][14], uint16_t Nid_cell);
void generate_dmrs_pbch(uint32_t dmrs_pbch_bitmap[DMRS_PBCH_I_SSB][DMRS_PBCH_N_HF][DMRS_BITMAP_SIZE], uint16_t Nid_cell);
uint16_t get_dmrs_freq_idx_ul(uint16_t n, uint8_t k_prime, uint8_t delta, uint8_t dmrs_type);

uint8_t allowed_xlsch_re_in_dmrs_symbol(uint16_t k,
                                        uint16_t start_sc,
                                        uint16_t ofdm_symbol_size,
                                        uint8_t numDmrsCdmGrpsNoData,
                                        uint8_t dmrs_type);

void nr_gen_ref_conj_symbols(uint32_t *in, uint32_t length, int16_t *output, uint16_t offset, int mod_order);
int8_t get_next_dmrs_symbol_in_slot(uint16_t  ul_dmrs_symb_pos, uint8_t counter, uint8_t end_symbol);
uint8_t get_dmrs_symbols_in_slot(uint16_t l_prime_mask,  uint16_t nb_symb);
int8_t get_valid_dmrs_idx_for_channel_est(uint16_t  dmrs_symb_pos, uint8_t counter);

static inline uint8_t is_dmrs_symbol(uint8_t l, uint16_t dmrsSymbMask ) { return ((dmrsSymbMask >> l) & 0x1); }
#undef EXTERN

#endif /* DMRS_NR_H */


