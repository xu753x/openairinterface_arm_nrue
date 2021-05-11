


#include "PHY/defs_eNB.h"
#ifdef DEBUG_UCI_TOOLS
  #include "PHY/vars.h"
#endif

//#define DEBUG_UCI 1


int16_t find_uci(uint16_t rnti, int frame, int subframe, PHY_VARS_eNB *eNB,find_type_t type) {
  uint16_t i;
  int16_t first_free_index=-1;
  AssertFatal(eNB!=NULL,"eNB is null\n");

  for (i=0; i<NUMBER_OF_UCI_VARS_MAX; i++) {
    if ((eNB->uci_vars[i].active >0) &&
        (eNB->uci_vars[i].rnti==rnti) &&
        (eNB->uci_vars[i].frame==frame) &&
        (eNB->uci_vars[i].subframe==subframe)) return(i);
    else if ((eNB->uci_vars[i].active == 0) && (first_free_index==-1)) first_free_index=i;
  }

  if (type == SEARCH_EXIST) return(-1);

  if (first_free_index==-1)
    LOG_E(MAC,"UCI table is full\n");

  return(first_free_index);
}



