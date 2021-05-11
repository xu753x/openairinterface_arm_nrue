



#ifndef NR_SCH_DMRS_H
#define NR_SCH_DMRS_H

#include "PHY/defs_nr_common.h"

#define NR_PDSCH_DMRS_ANTENNA_PORT0 1000
#define NR_PDSCH_DMRS_NB_ANTENNA_PORTS 12

void get_antenna_ports(uint8_t *ap, uint8_t n_symbs, uint8_t config);

void get_Wt(int8_t *Wt, uint8_t ap, uint8_t config);

void get_Wf(int8_t *Wf, uint8_t ap, uint8_t config);

uint8_t get_delta(uint8_t ap, uint8_t config);

uint16_t get_dmrs_freq_idx(uint16_t n, uint8_t k_prime, uint8_t delta, uint8_t dmrs_type);

uint8_t get_l0(uint16_t dlDmrsSymbPos);

#endif
