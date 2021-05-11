


//#include "PHY/defs.h"
#include <stdint.h>
#include "PHY/defs_nr_UE.h"
//#include "PHY/NR_TRANSPORT/nr_dci.h"
//#include "PHY/NR_UE_TRANSPORT/nr_transport_ue.h"
//#include "PHY/extern.h"
//#include "SCHED/defs.h"
#ifdef DEBUG_DCI_TOOLS
  #include "PHY/vars.h"
#endif
#include "assertions.h"

#include "SCHED_NR_UE/harq_nr.h"

//#define DEBUG_HARQ

//#include "LAYER2/MAC/extern.h"
//#include "LAYER2/MAC/defs.h"
//#include "../openair2/LAYER2/MAC/extern.h"
//#include "../openair2/LAYER2/MAC/defs.h"

//#define DEBUG_DCI
#define NR_PDCCH_DCI_TOOLS
//#define NR_PDCCH_DCI_TOOLS_DEBUG






uint8_t nr_subframe2harq_pid(NR_DL_FRAME_PARMS *frame_parms,uint32_t frame,uint8_t slot) {
  /*
    #ifdef DEBUG_DCI
    if (frame_parms->frame_type == TDD)
    printf("dci_tools.c: subframe2_harq_pid, subframe %d for TDD configuration %d\n",subframe,frame_parms->tdd_config);
    else
    printf("dci_tools.c: subframe2_harq_pid, subframe %d for FDD \n",subframe);
    #endif
  */
  uint8_t ret = 255;
  uint8_t subframe = slot / frame_parms->slots_per_subframe;

  AssertFatal(1==0,"Not ready for this ...\n");
  if (frame_parms->frame_type == FDD) {
    ret = (((frame<<1)+slot)&7);
  } else {

  }


  if (ret == 255) {
    LOG_E(PHY, "invalid harq_pid(%d) at SFN/SF = %d/%d\n", ret, frame, subframe);
    //mac_xface->macphy_exit("invalid harq_pid");
  }

  return ret;
}

uint8_t nr_pdcch_alloc2ul_subframe(NR_DL_FRAME_PARMS *frame_parms,uint8_t n) {

  AssertFatal(1==0,"Not ready for this\n");

}

uint32_t nr_pdcch_alloc2ul_frame(NR_DL_FRAME_PARMS *frame_parms,uint32_t frame, uint8_t n) {

  AssertFatal(1==0,"Not ready for this\n");

}
