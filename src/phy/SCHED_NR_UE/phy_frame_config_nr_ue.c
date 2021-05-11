



#include "PHY/defs_nr_UE.h"

/*******************************************************************
*
* NAME :         nr_ue_slot_select
*
* DESCRIPTION :  function for the UE equivalent to nr_slot_select
*
*********************************************************************/

int nr_ue_slot_select(fapi_nr_config_request_t *cfg, int nr_frame, int nr_slot) {
  /* for FFD all slot can be considered as an uplink */
  int mu = cfg->ssb_config.scs_common, check_slot = 0;

  if (cfg->cell_config.frame_duplex_type == FDD) {
    return (NR_UPLINK_SLOT | NR_DOWNLINK_SLOT);
  }
  if (cfg->tdd_table.max_tdd_periodicity_list == NULL) // this happens before receiving TDD configuration
    return (NR_DOWNLINK_SLOT);

  if (nr_frame%2 == 0) {
    for(int symbol_count=0; symbol_count<NR_NUMBER_OF_SYMBOLS_PER_SLOT; symbol_count++) {
      if (cfg->tdd_table.max_tdd_periodicity_list[nr_slot].max_num_of_symbol_per_slot_list[symbol_count].slot_config == 1) {
        check_slot++;
      }
    }

    if(check_slot == NR_NUMBER_OF_SYMBOLS_PER_SLOT) {
      return (NR_UPLINK_SLOT);
    }

    check_slot = 0;

    for(int symbol_count=0; symbol_count<NR_NUMBER_OF_SYMBOLS_PER_SLOT; symbol_count++) {
      if (cfg->tdd_table.max_tdd_periodicity_list[nr_slot].max_num_of_symbol_per_slot_list[symbol_count].slot_config == 0) {
        check_slot++;
      }
    }

    if(check_slot == NR_NUMBER_OF_SYMBOLS_PER_SLOT) {
      return (NR_DOWNLINK_SLOT);
    } else {
      return (NR_MIXED_SLOT);
    }
  } else {
    for(int symbol_count=0; symbol_count<NR_NUMBER_OF_SYMBOLS_PER_SLOT; symbol_count++) {
      if (cfg->tdd_table.max_tdd_periodicity_list[((1<<mu) * NR_NUMBER_OF_SUBFRAMES_PER_FRAME) + nr_slot].max_num_of_symbol_per_slot_list[symbol_count].slot_config == 1) {
        check_slot++;
      }
    }

    if(check_slot == NR_NUMBER_OF_SYMBOLS_PER_SLOT) {
      return (NR_UPLINK_SLOT);
    }

    check_slot = 0;

    for(int symbol_count=0; symbol_count<NR_NUMBER_OF_SYMBOLS_PER_SLOT; symbol_count++) {
      if (cfg->tdd_table.max_tdd_periodicity_list[((1<<mu) * NR_NUMBER_OF_SUBFRAMES_PER_FRAME) + nr_slot].max_num_of_symbol_per_slot_list[symbol_count].slot_config == 0) {
        check_slot++;
      }
    }

    if(check_slot == NR_NUMBER_OF_SYMBOLS_PER_SLOT) {
      return (NR_DOWNLINK_SLOT);
    } else {
      return (NR_MIXED_SLOT);
    }
  }
}
