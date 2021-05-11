




/** @addtogroup _PHY_TRANSPORT_
 * @{
 */

#ifndef __NR_TRANSPORT_COMMON_PROTO__H__
#define __NR_TRANSPORT_COMMON_PROTO__H__

#include "PHY/defs_nr_common.h"


void nr_group_sequence_hopping(pucch_GroupHopping_t PUCCH_GroupHopping,
                               uint32_t n_id,
                               uint8_t n_hop,
                               int nr_slot_tx,
                               uint8_t *u,
                               uint8_t *v);

double nr_cyclic_shift_hopping(uint32_t n_id,
                               uint8_t m0,
                               uint8_t mcs,
                               uint8_t lnormal,
                               uint8_t lprime,
                               int nr_slot_tx);

/** \brief Computes available bits G. */
uint32_t nr_get_G(uint16_t nb_rb, uint16_t nb_symb_sch, uint8_t nb_re_dmrs, uint16_t length_dmrs, uint8_t Qm, uint8_t Nl);

uint32_t nr_get_E(uint32_t G, uint8_t C, uint8_t Qm, uint8_t Nl, uint8_t r);

uint8_t nr_get_Qm_ul(uint8_t Imcs, uint8_t table_idx);

uint8_t nr_get_Qm_dl(uint8_t Imcs, uint8_t table_idx);

uint32_t nr_get_code_rate_ul(uint8_t Imcs, uint8_t table_idx);

uint32_t nr_get_code_rate_dl(uint8_t Imcs, uint8_t table_idx);

void compute_nr_prach_seq(uint8_t short_sequence,
                          uint8_t num_sequences,
                          uint8_t rootSequenceIndex,
                          uint32_t X_u[64][839]);

void nr_fill_du(uint16_t N_ZC,uint16_t *prach_root_sequence_map);

void init_nr_prach_tables(int N_ZC);

/**@}*/

void init_pucch2_luts(void);
#endif
