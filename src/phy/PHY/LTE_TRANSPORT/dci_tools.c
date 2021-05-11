



#include "PHY/defs_eNB.h"
#include "PHY/phy_extern.h"
#include "SCHED/sched_eNB.h"
#ifdef DEBUG_DCI_TOOLS
  #include "PHY/phy_vars.h"
#endif
#include "assertions.h"
#include "nfapi_interface.h"

//#define DEBUG_HARQ




#include "LAYER2/MAC/mac.h"

//#define DEBUG_DCI
#include "dci_tools_common_extern.h"
#include "transport_proto.h"

//#undef LOG_D
//#define LOG_D(A,B...) printf(B)

int16_t find_dlsch(uint16_t rnti, PHY_VARS_eNB *eNB,find_type_t type) {
  uint16_t i;
  int16_t first_free_index=-1;
  AssertFatal(eNB!=NULL,"eNB is null\n");

  for (i=0; i<NUMBER_OF_UE_MAX; i++) {
    AssertFatal(eNB->dlsch[i]!=NULL,"eNB->dlsch[%d] is null\n",i);
    AssertFatal(eNB->dlsch[i]!=NULL,"eNB->dlsch[%d][0] is null\n",i);
    LOG_D(PHY,"searching for rnti %x : UE index %d=> harq_mask %x, rnti %x, first_free_index %d\n", rnti,i,eNB->dlsch[i][0]->harq_mask,eNB->dlsch[i][0]->rnti,first_free_index);

    if ((eNB->dlsch[i][0]->harq_mask >0) &&
        (eNB->dlsch[i][0]->rnti==rnti))       return i;
    else if ((eNB->dlsch[i][0]->harq_mask == 0) && (first_free_index==-1)) first_free_index=i;
  }

  if (type == SEARCH_EXIST)
    return -1;

  if (first_free_index != -1)
    eNB->dlsch[first_free_index][0]->rnti = 0;

  return first_free_index;
}


int16_t find_ulsch(uint16_t rnti, PHY_VARS_eNB *eNB,find_type_t type) {
  uint16_t i;
  int16_t first_free_index=-1;
  AssertFatal(eNB!=NULL,"eNB is null\n");

  for (i=0; i<NUMBER_OF_UE_MAX; i++) {
    AssertFatal(eNB->ulsch[i]!=NULL,"eNB->ulsch[%d] is null\n",i);

    if ((eNB->ulsch[i]->harq_mask >0) &&
        (eNB->ulsch[i]->rnti==rnti))       return i;
    else if ((eNB->ulsch[i]->harq_mask == 0) && (first_free_index==-1)) first_free_index=i;
  }

  if (type == SEARCH_EXIST)
    return -1;

  if (first_free_index != -1)
    eNB->ulsch[first_free_index]->rnti = 0;

  return first_free_index;
}





void fill_pdcch_order(PHY_VARS_eNB *eNB,L1_rxtx_proc_t *proc,DCI_ALLOC_t *dci_alloc,nfapi_dl_config_dci_dl_pdu *pdu) {
  LTE_DL_FRAME_PARMS *fp = &eNB->frame_parms;
  uint8_t        *dci_pdu = &dci_alloc->dci_pdu[0];
  nfapi_dl_config_dci_dl_pdu_rel8_t *rel8 = &pdu->dci_dl_pdu_rel8;
  dci_alloc->firstCCE = rel8->cce_idx;
  dci_alloc->L = rel8->aggregation_level;
  dci_alloc->rnti = rel8->rnti;
  dci_alloc->harq_pid = rel8->harq_process;
  dci_alloc->ra_flag = 0;
  dci_alloc->format = format1A;
  LOG_D (PHY, "NFAPI: DCI format %d, nCCE %d, L %d, rnti %x,harq_pid %d\n", rel8->dci_format, rel8->cce_idx, rel8->aggregation_level, rel8->rnti, rel8->harq_process);

  switch (fp->N_RB_DL) {
    case 6:
      if (fp->frame_type == TDD) {
        dci_alloc->dci_length                         = sizeof_DCI1A_1_5MHz_TDD_1_6_t;
        ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->type     = 1;
        ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->vrb_type = rel8->virtual_resource_block_assignment_flag;
        ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->mcs      = rel8->mcs_1;
        ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->ndi      = rel8->new_data_indicator_1;
        ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
        ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->rv       = rel8->redundancy_version_1;
        ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->TPC      = rel8->tpc;
        ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->harq_pid = rel8->harq_process;
        ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->dai      = rel8->downlink_assignment_index;
        ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->padding  = 0;
      } else {
        dci_alloc->dci_length                         = sizeof_DCI1A_1_5MHz_FDD_t;
        ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->type         = 1;
        ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->vrb_type     = rel8->virtual_resource_block_assignment_flag;
        ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->mcs          = rel8->mcs_1;
        ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->ndi          = rel8->new_data_indicator_1;
        ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->rballoc      = rel8->resource_block_coding;
        ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->rv           = rel8->redundancy_version_1;
        ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->TPC          = rel8->tpc;
        ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->harq_pid     = rel8->harq_process;
        ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->padding      = 0;
        //      printf("FDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
      }

      break;

    case 25:
      if (fp->frame_type == TDD) {
        dci_alloc->dci_length                         = sizeof_DCI1A_5MHz_TDD_1_6_t;
        ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->type       = 1;
        ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->vrb_type   = rel8->virtual_resource_block_assignment_flag;
        ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->mcs        = rel8->mcs_1;
        ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->ndi        = rel8->new_data_indicator_1;
        ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->rballoc    = rel8->resource_block_coding;
        ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->rv         = rel8->redundancy_version_1;
        ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->TPC        = rel8->tpc;
        ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->harq_pid   = rel8->harq_process;
        ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->dai        = rel8->downlink_assignment_index;
        ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->padding    = 0;
        //        printf("TDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
      } else {
        dci_alloc->dci_length                         = sizeof_DCI1A_5MHz_FDD_t;
        ((DCI1A_5MHz_FDD_t *)dci_pdu)->type           = 1;
        ((DCI1A_5MHz_FDD_t *)dci_pdu)->vrb_type       = rel8->virtual_resource_block_assignment_flag;
        ((DCI1A_5MHz_FDD_t *)dci_pdu)->mcs            = rel8->mcs_1;
        ((DCI1A_5MHz_FDD_t *)dci_pdu)->ndi            = rel8->new_data_indicator_1;
        ((DCI1A_5MHz_FDD_t *)dci_pdu)->rballoc        = rel8->resource_block_coding;
        ((DCI1A_5MHz_FDD_t *)dci_pdu)->rv             = rel8->redundancy_version_1;
        ((DCI1A_5MHz_FDD_t *)dci_pdu)->TPC            = rel8->tpc;
        ((DCI1A_5MHz_FDD_t *)dci_pdu)->harq_pid       = rel8->harq_process;
        ((DCI1A_5MHz_FDD_t *)dci_pdu)->padding        = 0;
        //      printf("FDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
      }

      break;

    case 50:
      if (fp->frame_type == TDD) {
        dci_alloc->dci_length                         = sizeof_DCI1A_10MHz_TDD_1_6_t;
        ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->type      = 1;
        ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->vrb_type  = rel8->virtual_resource_block_assignment_flag;
        ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->mcs       = rel8->mcs_1;
        ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->ndi       = rel8->new_data_indicator_1;
        ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->rballoc   = rel8->resource_block_coding;
        ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->rv        = rel8->redundancy_version_1;
        ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->TPC       = rel8->tpc;
        ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->harq_pid  = rel8->harq_process;
        ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->dai       = rel8->downlink_assignment_index;
        ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->padding   = 0;
        //        printf("TDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
      } else {
        dci_alloc->dci_length                         = sizeof_DCI1A_10MHz_FDD_t;
        ((DCI1A_10MHz_FDD_t *)dci_pdu)->type          = 1;
        ((DCI1A_10MHz_FDD_t *)dci_pdu)->vrb_type      = rel8->virtual_resource_block_assignment_flag;
        ((DCI1A_10MHz_FDD_t *)dci_pdu)->mcs           = rel8->mcs_1;
        ((DCI1A_10MHz_FDD_t *)dci_pdu)->ndi           = rel8->new_data_indicator_1;
        ((DCI1A_10MHz_FDD_t *)dci_pdu)->rballoc       = rel8->resource_block_coding;
        ((DCI1A_10MHz_FDD_t *)dci_pdu)->rv            = rel8->redundancy_version_1;
        ((DCI1A_10MHz_FDD_t *)dci_pdu)->TPC           = rel8->tpc;
        ((DCI1A_10MHz_FDD_t *)dci_pdu)->harq_pid      = rel8->harq_process;
        ((DCI1A_10MHz_FDD_t *)dci_pdu)->padding       = 0;
        //      printf("FDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
      }

      break;

    case 100:
      if (fp->frame_type == TDD) {
        dci_alloc->dci_length                         = sizeof_DCI1A_20MHz_TDD_1_6_t;
        ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->type      = 1;
        ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->vrb_type  = rel8->virtual_resource_block_assignment_flag;
        ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->mcs       = rel8->mcs_1;
        ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->ndi       = rel8->new_data_indicator_1;
        ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->rballoc   = rel8->resource_block_coding;
        ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->rv        = rel8->redundancy_version_1;
        ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->TPC       = rel8->tpc;
        ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->harq_pid  = rel8->harq_process;
        ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->dai       = rel8->downlink_assignment_index;
        ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->padding   = 0;
        //        printf("TDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
      } else {
        dci_alloc->dci_length                         = sizeof_DCI1A_20MHz_FDD_t;
        ((DCI1A_20MHz_FDD_t *)dci_pdu)->type          = 1;
        ((DCI1A_20MHz_FDD_t *)dci_pdu)->vrb_type      = rel8->virtual_resource_block_assignment_flag;
        ((DCI1A_20MHz_FDD_t *)dci_pdu)->mcs           = rel8->mcs_1;
        ((DCI1A_20MHz_FDD_t *)dci_pdu)->ndi           = rel8->new_data_indicator_1;
        ((DCI1A_20MHz_FDD_t *)dci_pdu)->rballoc       = rel8->resource_block_coding;
        ((DCI1A_20MHz_FDD_t *)dci_pdu)->rv            = rel8->redundancy_version_1;
        ((DCI1A_20MHz_FDD_t *)dci_pdu)->TPC           = rel8->tpc;
        ((DCI1A_20MHz_FDD_t *)dci_pdu)->harq_pid      = rel8->harq_process;
        ((DCI1A_20MHz_FDD_t *)dci_pdu)->padding       = 0;
        //      printf("FDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
      }

      break;
  }

  LOG_T(PHY,"%d.%d: DCI 1A: rnti %x, PDCCH order to do PRACH\n",
        proc->frame_tx, proc->subframe_tx, rel8->rnti);
}

void fill_dci_and_dlsch(PHY_VARS_eNB *eNB,
                        int frame,
                        int subframe,
                        L1_rxtx_proc_t *proc,
                        DCI_ALLOC_t *dci_alloc,
                        nfapi_dl_config_dci_dl_pdu *pdu) {
  LTE_DL_FRAME_PARMS *fp = &eNB->frame_parms;
  uint8_t        *dci_pdu = &dci_alloc->dci_pdu[0];
  nfapi_dl_config_dci_dl_pdu_rel8_t *rel8 = &pdu->dci_dl_pdu_rel8;

  /* check if this is a DCI 1A PDCCH order for RAPROC */
  if (rel8->dci_format == NFAPI_DL_DCI_FORMAT_1A && rel8->rnti_type == 1) {
    int             full_rb;

    switch (fp->N_RB_DL) {
      case 6:
        full_rb = 63;
        break;

      case 25:
        full_rb = 511;
        break;

      case 50:
        full_rb = 2047;
        break;

      case 100:
        full_rb = 8191;
        break;

      default:
        abort ();
    }

    if (rel8->resource_block_coding == full_rb)
      return fill_pdcch_order (eNB, proc, dci_alloc, pdu);
  }

  LTE_eNB_DLSCH_t *dlsch0=NULL,*dlsch1=NULL;
  LTE_DL_eNB_HARQ_t *dlsch0_harq=NULL,*dlsch1_harq=NULL;
  int beamforming_mode = 0;
  int UE_id=-1;
  int NPRB;
  int TB0_active;
  int TB1_active;
  uint16_t DL_pmi_single=0; // This should be taken from DLSCH parameters for PUSCH precoding
  uint8_t I_mcs = 0;
  dci_alloc->firstCCE = rel8->cce_idx;
  dci_alloc->L = rel8->aggregation_level;
  dci_alloc->rnti = rel8->rnti;
  dci_alloc->harq_pid = rel8->harq_process;
  dci_alloc->ra_flag = 0;
  LOG_D(PHY,"NFAPI: SFN/SF:%04d%d proc:TX:[SFN/SF:%04d%d] DCI format %d, nCCE %d, L %d, rnti %x, harq_pid %d\n",
        frame,subframe,proc->frame_tx,proc->subframe_tx,rel8->dci_format,rel8->cce_idx,rel8->aggregation_level,rel8->rnti,rel8->harq_process);

  if ((rel8->rnti_type == 2 ) && (rel8->rnti != SI_RNTI) && (rel8->rnti != P_RNTI)) dci_alloc->ra_flag = 1;

  UE_id = find_dlsch(rel8->rnti,eNB,SEARCH_EXIST_OR_FREE);

  if( (UE_id<0) || (UE_id>=NUMBER_OF_UE_MAX) ) {
    LOG_E(PHY,"illegal UE_id found!!! rnti %04x UE_id %d\n",rel8->rnti,UE_id);
    return;
  }

  //AssertFatal(UE_id!=-1,"no free or exiting dlsch_context\n");
  //AssertFatal(UE_id<NUMBER_OF_UE_MAX,"returned UE_id %d >= %d(NUMBER_OF_UE_MAX)\n",UE_id,NUMBER_OF_UE_MAX);
  dlsch0 = eNB->dlsch[UE_id][0];
  dlsch1 = eNB->dlsch[UE_id][1];
  dlsch0->ue_type = 0;
  dlsch1->ue_type = 0;
  beamforming_mode                          = eNB->transmission_mode[(uint8_t)UE_id]<7?0:eNB->transmission_mode[(uint8_t)UE_id];
  dlsch0_harq                               = dlsch0->harq_processes[rel8->harq_process];
  dlsch0_harq->codeword                     = 0;
  dlsch1_harq                               = dlsch1->harq_processes[rel8->harq_process];
  dlsch1_harq->codeword                     = 1;
  dlsch0->subframe_tx[subframe]             = 1;
  LOG_D(PHY,"NFAPI: SFN/SF:%04d%d proc:TX:SFN/SF:%04d%d dlsch0[rnti:%x harq_mask:%04x] dci_pdu[rnti:%x rnti_type:%d harq_process:%d ndi1:%d] dlsch0_harq[round:%d harq_mask:%x ndi:%d]\n",
        frame,subframe,
        proc->frame_tx,proc->subframe_tx,
        dlsch0->rnti,dlsch0->harq_mask,
        rel8->rnti, rel8->rnti_type, rel8->harq_process, rel8->new_data_indicator_1,
        dlsch0_harq->round, dlsch0->harq_mask, dlsch0_harq->ndi);

  if (dlsch0->rnti != rel8->rnti) { // if rnti of dlsch is not the same as in the config, this is a new entry
    dlsch0_harq->round=0;
    dlsch0->harq_mask=0;
  }

  if ((dlsch0->harq_mask & (1 << rel8->harq_process)) > 0) {
    if (rel8->new_data_indicator_1 != dlsch0_harq->ndi)
      dlsch0_harq->round = 0;
  } else {                      // process is inactive, so activate and set round to 0
    dlsch0_harq->round = 0;
  }

  dlsch0_harq->ndi = rel8->new_data_indicator_1;
#ifdef PHY_TX_THREAD
  dlsch0->active[subframe]        = 1;
#else
  dlsch0->active        = 1;
#endif

  if (rel8->rnti_type == 2)
    dlsch0_harq->round    = 0;

  LOG_D(PHY,"NFAPI: rel8[rnti %x dci_format %d harq_process %d ndi1 %d rnti type %d] dlsch0[rnti %x harq_mask %x] dlsch0_harq[round %d ndi %d]\n",
        rel8->rnti,rel8->dci_format,rel8->harq_process, rel8->new_data_indicator_1, rel8->rnti_type,
        dlsch0->rnti,dlsch0->harq_mask,
        dlsch0_harq->round,dlsch0_harq->ndi
       );

  switch (rel8->dci_format) {
    case NFAPI_DL_DCI_FORMAT_1A:
      AssertFatal(rel8->resource_block_coding < 8192, "SFN/SF:%04d%d proc:TX:SFN/SF:%04d%d: rel8->resource_block_coding (%p) %u >= 8192 (rnti %x, rnti_type %d, format %d, harq_id %d\n",
                  frame,subframe,proc->frame_tx,subframe,
                  &rel8->resource_block_coding,rel8->resource_block_coding,rel8->rnti,rel8->rnti_type,rel8->dci_format,rel8->harq_process);
      dci_alloc->format = format1A;

      switch (fp->N_RB_DL) {
        case 6:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                         = sizeof_DCI1A_1_5MHz_TDD_1_6_t;
            ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->type     = 1;
            ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->vrb_type = rel8->virtual_resource_block_assignment_flag;
            ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->mcs      = rel8->mcs_1;
            ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->ndi      = rel8->new_data_indicator_1;
            ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
            ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->rv       = rel8->redundancy_version_1;
            ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->TPC      = rel8->tpc;
            ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->harq_pid = rel8->harq_process;
            ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->dai      = rel8->downlink_assignment_index;
            ((DCI1A_1_5MHz_TDD_1_6_t *)dci_pdu)->padding  = 0;
          } else {
            dci_alloc->dci_length                         = sizeof_DCI1A_1_5MHz_FDD_t;
            ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->type         = 1;
            ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->vrb_type     = rel8->virtual_resource_block_assignment_flag;
            ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->mcs          = rel8->mcs_1;
            ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->ndi          = rel8->new_data_indicator_1;
            ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->rballoc      = rel8->resource_block_coding;
            ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->rv           = rel8->redundancy_version_1;
            ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->TPC          = rel8->tpc;
            ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->harq_pid     = rel8->harq_process;
            ((DCI1A_1_5MHz_FDD_t *)dci_pdu)->padding      = 0;
            //      printf("FDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          }

          AssertFatal (rel8->virtual_resource_block_assignment_flag == LOCALIZED, "Distributed RB allocation not done yet\n");
          dlsch0_harq->rb_alloc[0] = localRIV2alloc_LUT6[rel8->resource_block_coding];
          dlsch0_harq->vrb_type = rel8->virtual_resource_block_assignment_flag;
          dlsch0_harq->nb_rb = RIV2nb_rb_LUT6[rel8->resource_block_coding]; //NPRB;
          break;

        case 25:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                         = sizeof_DCI1A_5MHz_TDD_1_6_t;
            ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->type       = 1;
            ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->vrb_type   = rel8->virtual_resource_block_assignment_flag;
            ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->mcs        = rel8->mcs_1;
            ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->ndi        = rel8->new_data_indicator_1;
            ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->rballoc    = rel8->resource_block_coding;
            ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->rv         = rel8->redundancy_version_1;
            ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->TPC        = rel8->tpc;
            ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->harq_pid   = rel8->harq_process;
            ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->dai        = rel8->downlink_assignment_index;
            ((DCI1A_5MHz_TDD_1_6_t *)dci_pdu)->padding    = 0;
            //        printf("TDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                         = sizeof_DCI1A_5MHz_FDD_t;
            ((DCI1A_5MHz_FDD_t *)dci_pdu)->type           = 1;
            ((DCI1A_5MHz_FDD_t *)dci_pdu)->vrb_type       = rel8->virtual_resource_block_assignment_flag;
            ((DCI1A_5MHz_FDD_t *)dci_pdu)->mcs            = rel8->mcs_1;
            ((DCI1A_5MHz_FDD_t *)dci_pdu)->ndi            = rel8->new_data_indicator_1;
            ((DCI1A_5MHz_FDD_t *)dci_pdu)->rballoc        = rel8->resource_block_coding;
            ((DCI1A_5MHz_FDD_t *)dci_pdu)->rv             = rel8->redundancy_version_1;
            ((DCI1A_5MHz_FDD_t *)dci_pdu)->TPC            = rel8->tpc;
            ((DCI1A_5MHz_FDD_t *)dci_pdu)->harq_pid       = rel8->harq_process;
            ((DCI1A_5MHz_FDD_t *)dci_pdu)->padding        = 0;
            //      printf("FDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          }

          AssertFatal (rel8->virtual_resource_block_assignment_flag == LOCALIZED, "Distributed RB allocation not done yet\n");
          dlsch0_harq->rb_alloc[0] = localRIV2alloc_LUT25[rel8->resource_block_coding];
          dlsch0_harq->vrb_type = rel8->virtual_resource_block_assignment_flag;
          dlsch0_harq->nb_rb = RIV2nb_rb_LUT25[rel8->resource_block_coding];        //NPRB;
          break;

        case 50:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                         = sizeof_DCI1A_10MHz_TDD_1_6_t;
            ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->type      = 1;
            ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->vrb_type  = rel8->virtual_resource_block_assignment_flag;
            ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->mcs       = rel8->mcs_1;
            ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->ndi       = rel8->new_data_indicator_1;
            ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->rballoc   = rel8->resource_block_coding;
            ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->rv        = rel8->redundancy_version_1;
            ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->TPC       = rel8->tpc;
            ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->harq_pid  = rel8->harq_process;
            ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->dai       = rel8->downlink_assignment_index;
            ((DCI1A_10MHz_TDD_1_6_t *)dci_pdu)->padding   = 0;
            //        printf("TDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                         = sizeof_DCI1A_10MHz_FDD_t;
            ((DCI1A_10MHz_FDD_t *)dci_pdu)->type          = 1;
            ((DCI1A_10MHz_FDD_t *)dci_pdu)->vrb_type      = rel8->virtual_resource_block_assignment_flag;
            ((DCI1A_10MHz_FDD_t *)dci_pdu)->mcs           = rel8->mcs_1;
            ((DCI1A_10MHz_FDD_t *)dci_pdu)->ndi           = rel8->new_data_indicator_1;
            ((DCI1A_10MHz_FDD_t *)dci_pdu)->rballoc       = rel8->resource_block_coding;
            ((DCI1A_10MHz_FDD_t *)dci_pdu)->rv            = rel8->redundancy_version_1;
            ((DCI1A_10MHz_FDD_t *)dci_pdu)->TPC           = rel8->tpc;
            ((DCI1A_10MHz_FDD_t *)dci_pdu)->harq_pid      = rel8->harq_process;
            ((DCI1A_10MHz_FDD_t *)dci_pdu)->padding       = 0;
            //      printf("FDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          }

          AssertFatal (rel8->virtual_resource_block_assignment_flag == LOCALIZED, "Distributed RB allocation not done yet\n");
          dlsch0_harq->rb_alloc[0] = localRIV2alloc_LUT50_0[rel8->resource_block_coding];
          dlsch0_harq->rb_alloc[1] = localRIV2alloc_LUT50_1[rel8->resource_block_coding];
          dlsch0_harq->vrb_type = rel8->virtual_resource_block_assignment_flag;
          dlsch0_harq->nb_rb = RIV2nb_rb_LUT50[rel8->resource_block_coding];        //NPRB;
          break;

        case 100:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                         = sizeof_DCI1A_20MHz_TDD_1_6_t;
            ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->type      = 1;
            ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->vrb_type  = rel8->virtual_resource_block_assignment_flag;
            ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->mcs       = rel8->mcs_1;
            ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->ndi       = rel8->new_data_indicator_1;
            ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->rballoc   = rel8->resource_block_coding;
            ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->rv        = rel8->redundancy_version_1;
            ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->TPC       = rel8->tpc;
            ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->harq_pid  = rel8->harq_process;
            ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->dai       = rel8->downlink_assignment_index;
            ((DCI1A_20MHz_TDD_1_6_t *)dci_pdu)->padding   = 0;
            //        printf("TDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                         = sizeof_DCI1A_20MHz_FDD_t;
            ((DCI1A_20MHz_FDD_t *)dci_pdu)->type          = 1;
            ((DCI1A_20MHz_FDD_t *)dci_pdu)->vrb_type      = rel8->virtual_resource_block_assignment_flag;
            ((DCI1A_20MHz_FDD_t *)dci_pdu)->mcs           = rel8->mcs_1;
            ((DCI1A_20MHz_FDD_t *)dci_pdu)->ndi           = rel8->new_data_indicator_1;
            ((DCI1A_20MHz_FDD_t *)dci_pdu)->rballoc       = rel8->resource_block_coding;
            ((DCI1A_20MHz_FDD_t *)dci_pdu)->rv            = rel8->redundancy_version_1;
            ((DCI1A_20MHz_FDD_t *)dci_pdu)->TPC           = rel8->tpc;
            ((DCI1A_20MHz_FDD_t *)dci_pdu)->harq_pid      = rel8->harq_process;
            ((DCI1A_20MHz_FDD_t *)dci_pdu)->padding       = 0;
            //      printf("FDD 1A: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          }

          AssertFatal (rel8->virtual_resource_block_assignment_flag == LOCALIZED, "Distributed RB allocation not done yet\n");
          dlsch0_harq->rb_alloc[0] = localRIV2alloc_LUT100_0[rel8->resource_block_coding];
          dlsch0_harq->rb_alloc[1] = localRIV2alloc_LUT100_1[rel8->resource_block_coding];
          dlsch0_harq->rb_alloc[2] = localRIV2alloc_LUT100_2[rel8->resource_block_coding];
          dlsch0_harq->rb_alloc[3] = localRIV2alloc_LUT100_3[rel8->resource_block_coding];
          dlsch0_harq->vrb_type = rel8->virtual_resource_block_assignment_flag;
          dlsch0_harq->nb_rb = RIV2nb_rb_LUT100[rel8->resource_block_coding];       //NPRB;
          break;
      }

      if (rel8->rnti_type == 2) {
        // see 36-212 V8.6.0 p. 45
        NPRB = (rel8->tpc & 1) + 2;
        // 36-213 sec.7.1.7.2 p.26
        I_mcs     = rel8->mcs_1;
      } else {
        NPRB      = dlsch0_harq->nb_rb;
        I_mcs     = get_I_TBS(rel8->mcs_1);
      }

      AssertFatal(NPRB>0,"DCI 1A: NPRB = 0 (rnti %x, rnti type %d, tpc %d, round %d, resource_block_coding %d, harq process %d)\n",rel8->rnti,rel8->rnti_type,rel8->tpc,dlsch0_harq->round,
                  rel8->resource_block_coding,rel8->harq_process);
      dlsch0_harq->rvidx         = rel8->redundancy_version_1;
      dlsch0_harq->Nl            = 1;
      dlsch0_harq->mimo_mode     = (fp->nb_antenna_ports_eNB == 1) ? SISO : ALAMOUTI;
      dlsch0_harq->dl_power_off  = 1;
      dlsch0_harq->mcs             = rel8->mcs_1;
      dlsch0_harq->Qm              = 2;
      dlsch0_harq->TBS             = TBStable[I_mcs][NPRB-1];
      dlsch0->harq_ids[frame%2][subframe]   = rel8->harq_process;
#ifdef PHY_TX_THREAD
      dlsch0->active[subframe]     = 1;
#else
      dlsch0->active               = 1;
#endif
      dlsch0->rnti                 = rel8->rnti;

      //dlsch0->harq_ids[subframe]   = rel8->harq_process;

      if (dlsch0_harq->round == 0)
        dlsch0_harq->status = ACTIVE;

      dlsch0->harq_mask |= (1 << rel8->harq_process);

      if (rel8->rnti_type == 1) LOG_D(PHY,"DCI 1A: round %d, mcs %d, TBS %d, rballoc %x, rv %d, rnti %x, harq process %d\n",dlsch0_harq->round,rel8->mcs_1,dlsch0_harq->TBS,rel8->resource_block_coding,
                                        rel8->redundancy_version_1,rel8->rnti,rel8->harq_process);

      break;

    case NFAPI_DL_DCI_FORMAT_1:
      dci_alloc->format           = format1;
#ifdef PHY_TX_THREAD
      dlsch0->active[subframe]    = 1;
#else
      dlsch0->active              = 1;
#endif
      LOG_D(PHY,"SFN/SF:%04d%d proc:TX:SFN/SF:%04d%d: Programming DLSCH for Format 1 DCI, harq_pid %d\n",frame,subframe,proc->frame_tx,subframe,rel8->harq_process);

      switch (fp->N_RB_DL) {
        case 6:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                     = sizeof_DCI1_1_5MHz_TDD_t;
            ((DCI1_1_5MHz_TDD_t *)dci_pdu)->rah       = rel8->resource_allocation_type;
            ((DCI1_1_5MHz_TDD_t *)dci_pdu)->mcs       = rel8->mcs_1;
            ((DCI1_1_5MHz_TDD_t *)dci_pdu)->ndi       = rel8->new_data_indicator_1;
            ((DCI1_1_5MHz_TDD_t *)dci_pdu)->rballoc   = rel8->resource_block_coding;
            ((DCI1_1_5MHz_TDD_t *)dci_pdu)->rv        = rel8->redundancy_version_1;
            ((DCI1_1_5MHz_TDD_t *)dci_pdu)->TPC       = rel8->tpc;
            ((DCI1_1_5MHz_TDD_t *)dci_pdu)->harq_pid  = rel8->harq_process;
            ((DCI1_1_5MHz_TDD_t *)dci_pdu)->dai       = rel8->downlink_assignment_index;
            ((DCI1_1_5MHz_TDD_t *)dci_pdu)->padding   = 0;
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                         = sizeof_DCI1_1_5MHz_FDD_t;
            ((DCI1_1_5MHz_FDD_t *)dci_pdu)->rah           = rel8->resource_allocation_type;
            ((DCI1_1_5MHz_FDD_t *)dci_pdu)->mcs           = rel8->mcs_1;
            ((DCI1_1_5MHz_FDD_t *)dci_pdu)->ndi       = rel8->new_data_indicator_1;
            ((DCI1_1_5MHz_FDD_t *)dci_pdu)->rballoc       = rel8->resource_block_coding;
            ((DCI1_1_5MHz_FDD_t *)dci_pdu)->rv            = rel8->redundancy_version_1;
            ((DCI1_1_5MHz_FDD_t *)dci_pdu)->TPC           = rel8->tpc;
            ((DCI1_1_5MHz_FDD_t *)dci_pdu)->harq_pid      = rel8->harq_process;
            ((DCI1_1_5MHz_FDD_t *)dci_pdu)->padding       = 0;
            //      printf("FDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          }

          break;

        case 25:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                      = sizeof_DCI1_5MHz_TDD_t;
            ((DCI1_5MHz_TDD_t *)dci_pdu)->rah      = rel8->resource_allocation_type;
            ((DCI1_5MHz_TDD_t *)dci_pdu)->mcs      = rel8->mcs_1;
            ((DCI1_5MHz_TDD_t *)dci_pdu)->ndi      = rel8->new_data_indicator_1;
            ((DCI1_5MHz_TDD_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
            ((DCI1_5MHz_TDD_t *)dci_pdu)->rv       = rel8->redundancy_version_1;
            ((DCI1_5MHz_TDD_t *)dci_pdu)->TPC      = rel8->tpc;
            ((DCI1_5MHz_TDD_t *)dci_pdu)->harq_pid = rel8->harq_process;
            ((DCI1_5MHz_TDD_t *)dci_pdu)->dai      = rel8->downlink_assignment_index;
            ((DCI1_5MHz_TDD_t *)dci_pdu)->padding  = 0;
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                  = sizeof_DCI1_5MHz_FDD_t;
            ((DCI1_5MHz_FDD_t *)dci_pdu)->rah      = rel8->resource_allocation_type;
            ((DCI1_5MHz_FDD_t *)dci_pdu)->mcs      = rel8->mcs_1;
            ((DCI1_5MHz_FDD_t *)dci_pdu)->ndi      = rel8->new_data_indicator_1;
            ((DCI1_5MHz_FDD_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
            ((DCI1_5MHz_FDD_t *)dci_pdu)->rv       = rel8->redundancy_version_1;
            ((DCI1_5MHz_FDD_t *)dci_pdu)->TPC      = rel8->tpc;
            ((DCI1_5MHz_FDD_t *)dci_pdu)->harq_pid = rel8->harq_process;
            ((DCI1_5MHz_FDD_t *)dci_pdu)->padding  = 0;
            //      printf("FDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          }

          break;

        case 50:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                       = sizeof_DCI1_10MHz_TDD_t;
            ((DCI1_10MHz_TDD_t *)dci_pdu)->rah      = rel8->resource_allocation_type;
            ((DCI1_10MHz_TDD_t *)dci_pdu)->mcs      = rel8->mcs_1;
            ((DCI1_10MHz_TDD_t *)dci_pdu)->ndi      = rel8->new_data_indicator_1;
            ((DCI1_10MHz_TDD_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
            ((DCI1_10MHz_TDD_t *)dci_pdu)->rv       = rel8->redundancy_version_1;
            ((DCI1_10MHz_TDD_t *)dci_pdu)->TPC      = rel8->tpc;
            ((DCI1_10MHz_TDD_t *)dci_pdu)->harq_pid = rel8->harq_process;
            ((DCI1_10MHz_TDD_t *)dci_pdu)->dai      = rel8->downlink_assignment_index;
            ((DCI1_10MHz_TDD_t *)dci_pdu)->padding  = 0;
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                   = sizeof_DCI1_10MHz_FDD_t;
            ((DCI1_10MHz_FDD_t *)dci_pdu)->rah      = rel8->resource_allocation_type;
            ((DCI1_10MHz_FDD_t *)dci_pdu)->mcs      = rel8->mcs_1;
            ((DCI1_10MHz_FDD_t *)dci_pdu)->ndi      = rel8->new_data_indicator_1;
            ((DCI1_10MHz_FDD_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
            ((DCI1_10MHz_FDD_t *)dci_pdu)->rv       = rel8->redundancy_version_1;
            ((DCI1_10MHz_FDD_t *)dci_pdu)->TPC      = rel8->tpc;
            ((DCI1_10MHz_FDD_t *)dci_pdu)->harq_pid = rel8->harq_process;
            ((DCI1_10MHz_FDD_t *)dci_pdu)->padding  = 0;
          }

          break;

        case 100:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                       = sizeof_DCI1_20MHz_TDD_t;
            ((DCI1_20MHz_TDD_t *)dci_pdu)->rah      = rel8->resource_allocation_type;
            ((DCI1_20MHz_TDD_t *)dci_pdu)->mcs      = rel8->mcs_1;
            ((DCI1_20MHz_TDD_t *)dci_pdu)->ndi      = rel8->new_data_indicator_1;
            ((DCI1_20MHz_TDD_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
            ((DCI1_20MHz_TDD_t *)dci_pdu)->rv       = rel8->redundancy_version_1;
            ((DCI1_20MHz_TDD_t *)dci_pdu)->TPC      = rel8->tpc;
            ((DCI1_20MHz_TDD_t *)dci_pdu)->harq_pid = rel8->harq_process;
            ((DCI1_20MHz_TDD_t *)dci_pdu)->dai      = rel8->downlink_assignment_index;
            ((DCI1_20MHz_TDD_t *)dci_pdu)->padding  = 0;
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                   = sizeof_DCI1_20MHz_FDD_t;
            ((DCI1_20MHz_FDD_t *)dci_pdu)->rah      = rel8->resource_allocation_type;
            ((DCI1_20MHz_FDD_t *)dci_pdu)->mcs      = rel8->mcs_1;
            ((DCI1_20MHz_FDD_t *)dci_pdu)->ndi      = rel8->new_data_indicator_1;
            ((DCI1_20MHz_FDD_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
            ((DCI1_20MHz_FDD_t *)dci_pdu)->rv       = rel8->redundancy_version_1;
            ((DCI1_20MHz_FDD_t *)dci_pdu)->TPC      = rel8->tpc;
            ((DCI1_20MHz_FDD_t *)dci_pdu)->harq_pid = rel8->harq_process;
            ((DCI1_20MHz_FDD_t *)dci_pdu)->padding  = 0;
          }

          break;
      }

      AssertFatal (rel8->harq_process < 8, "Format 1: harq_pid=%d >= 8\n", rel8->harq_process);
      dlsch0_harq = dlsch0->harq_processes[rel8->harq_process];
      dlsch0_harq->codeword = 0;
      // printf("DCI: Setting subframe_tx for subframe %d\n",subframe);
      dlsch0->subframe_tx[subframe] = 1;
      conv_rballoc (rel8->resource_allocation_type, rel8->resource_block_coding, fp->N_RB_DL, dlsch0_harq->rb_alloc);
      dlsch0_harq->nb_rb = conv_nprb (rel8->resource_allocation_type, rel8->resource_block_coding, fp->N_RB_DL);
      NPRB = dlsch0_harq->nb_rb;
      AssertFatal (NPRB > 0, "NPRB == 0\n");
      dlsch0_harq->rvidx = rel8->redundancy_version_1;
      dlsch0_harq->Nl = 1;

      //    dlsch[0]->layer_index = 0;
      if (beamforming_mode == 0)
        dlsch0_harq->mimo_mode = (fp->nb_antenna_ports_eNB == 1) ? SISO : ALAMOUTI;
      else if (beamforming_mode == 7)
        dlsch0_harq->mimo_mode = TM7;
      else
        LOG_E (PHY, "Invalid beamforming mode %dL\n", beamforming_mode);

      dlsch0_harq->dl_power_off = 1;
#ifdef PHY_TX_THREAD
      dlsch0->active[subframe] = 1;
#else
      dlsch0->active = 1;
#endif

      if (dlsch0_harq->round == 0) {
        dlsch0_harq->status = ACTIVE;
        //            printf("Setting DLSCH process %d to ACTIVE\n",rel8->harq_process);
        // MCS and TBS don't change across HARQ rounds
        dlsch0_harq->mcs = rel8->mcs_1;
        dlsch0_harq->Qm = get_Qm (rel8->mcs_1);
        dlsch0_harq->TBS = TBStable[get_I_TBS (dlsch0_harq->mcs)][NPRB - 1];
      }

      LOG_D(PHY,"DCI: Set harq_ids[%d] to %d (%p)\n",subframe,rel8->harq_process,dlsch0);
      dlsch0->harq_ids[frame%2][subframe] = rel8->harq_process;
      dlsch0->harq_mask |= (1 << rel8->harq_process);
      dlsch0->rnti = rel8->rnti;
      break;

    case NFAPI_DL_DCI_FORMAT_2A:
      dci_alloc->format = format2A;

      switch (fp->N_RB_DL) {
        case 6:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                         = sizeof_DCI2A_1_5MHz_2A_TDD_t;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->rah      = rel8->resource_allocation_type;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->mcs1     = rel8->mcs_1;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->mcs2     = rel8->mcs_2;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->ndi1     = rel8->new_data_indicator_1;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->ndi2     = rel8->new_data_indicator_2;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->rv1      = rel8->redundancy_version_1;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->rv2      = rel8->redundancy_version_2;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->TPC      = rel8->tpc;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->harq_pid = rel8->harq_process;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->tb_swap  = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2A_1_5MHz_2A_TDD_t *)dci_pdu)->dai      = rel8->downlink_assignment_index;
            /* there is no padding in this structure, it is exactly 32 bits */
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                         = sizeof_DCI2A_1_5MHz_2A_FDD_t;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->rah          = rel8->resource_allocation_type;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->mcs1         = rel8->mcs_1;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->mcs2         = rel8->mcs_2;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->ndi1         = rel8->new_data_indicator_1;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->ndi2         = rel8->new_data_indicator_2;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->rballoc      = rel8->resource_block_coding;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->rv1          = rel8->redundancy_version_1;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->rv2          = rel8->redundancy_version_2;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->TPC          = rel8->tpc;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->harq_pid     = rel8->harq_process;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->tb_swap      = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2A_1_5MHz_2A_FDD_t *)dci_pdu)->padding      = 0;
            //      printf("FDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          }

          break;

        case 25:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                        = sizeof_DCI2A_5MHz_2A_TDD_t;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->rah       = rel8->resource_allocation_type;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->mcs1      = rel8->mcs_1;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->mcs2      = rel8->mcs_2;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->ndi1      = rel8->new_data_indicator_1;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->ndi2      = rel8->new_data_indicator_2;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->rballoc   = rel8->resource_block_coding;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->rv1       = rel8->redundancy_version_1;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->rv2       = rel8->redundancy_version_2;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->TPC       = rel8->tpc;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->harq_pid  = rel8->harq_process;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->tb_swap   = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->dai       = rel8->downlink_assignment_index;
            ((DCI2A_5MHz_2A_TDD_t *)dci_pdu)->padding   = 0;
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                           = sizeof_DCI2A_5MHz_2A_FDD_t;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->rah           = rel8->resource_allocation_type;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->mcs1          = rel8->mcs_1;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->mcs2          = rel8->mcs_2;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->ndi1          = rel8->new_data_indicator_1;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->ndi2          = rel8->new_data_indicator_2;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->rballoc       = rel8->resource_block_coding;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->rv1           = rel8->redundancy_version_1;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->rv2           = rel8->redundancy_version_2;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->TPC           = rel8->tpc;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->harq_pid      = rel8->harq_process;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->tb_swap       = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2A_5MHz_2A_FDD_t *)dci_pdu)->padding       = 0;
            //      printf("FDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          }

          break;

        case 50:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                        = sizeof_DCI2A_10MHz_2A_TDD_t;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->rah      = rel8->resource_allocation_type;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->mcs1     = rel8->mcs_1;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->mcs2     = rel8->mcs_2;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->ndi1     = rel8->new_data_indicator_1;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->ndi2     = rel8->new_data_indicator_2;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->rv1      = rel8->redundancy_version_1;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->rv2      = rel8->redundancy_version_2;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->TPC      = rel8->tpc;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->harq_pid = rel8->harq_process;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->tb_swap  = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->dai      = rel8->downlink_assignment_index;
            ((DCI2A_10MHz_2A_TDD_t *)dci_pdu)->padding  = 0;
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                        = sizeof_DCI2A_10MHz_2A_FDD_t;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->rah       = rel8->resource_allocation_type;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->mcs1      = rel8->mcs_1;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->mcs2      = rel8->mcs_2;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->ndi1      = rel8->new_data_indicator_1;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->ndi2      = rel8->new_data_indicator_2;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->rballoc   = rel8->resource_block_coding;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->rv1       = rel8->redundancy_version_1;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->rv2       = rel8->redundancy_version_2;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->TPC       = rel8->tpc;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->harq_pid  = rel8->harq_process;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->tb_swap   = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2A_10MHz_2A_FDD_t *)dci_pdu)->padding   = 0;
          }

          break;

        case 100:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                        = sizeof_DCI2A_20MHz_2A_TDD_t;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->rah       = rel8->resource_allocation_type;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->mcs1      = rel8->mcs_1;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->mcs2      = rel8->mcs_2;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->ndi1      = rel8->new_data_indicator_1;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->ndi2      = rel8->new_data_indicator_2;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->rballoc   = rel8->resource_block_coding;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->rv1       = rel8->redundancy_version_1;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->rv2       = rel8->redundancy_version_2;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->TPC       = rel8->tpc;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->harq_pid  = rel8->harq_process;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->tb_swap   = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->dai       = rel8->downlink_assignment_index;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->padding   = 0;
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                           = sizeof_DCI2A_20MHz_2A_FDD_t;
            ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->rah          = rel8->resource_allocation_type;
            ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->mcs1         = rel8->mcs_1;
            ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->mcs2         = rel8->mcs_2;
            ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->ndi1         = rel8->new_data_indicator_1;
            ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->ndi2         = rel8->new_data_indicator_2;
            ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->rballoc      = rel8->resource_block_coding;
            ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->rv1          = rel8->redundancy_version_1;
            ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->rv2          = rel8->redundancy_version_2;
            ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->TPC          = rel8->tpc;
            ((DCI2A_20MHz_2A_FDD_t *)dci_pdu)->harq_pid     = rel8->harq_process;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->tb_swap      = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2A_20MHz_2A_TDD_t *)dci_pdu)->padding      = 0;
          }

          break;
      }

      AssertFatal (rel8->harq_process < 8, "Format 2_2A: harq_pid=%d >= 8\n", rel8->harq_process);

      // Flip the TB to codeword mapping as described in 5.3.3.1.5 of 36-212 V11.3.0
      // note that we must set tbswap=0 in eNB scheduler if one TB is deactivated

      // This must be set as in TM4, does not work properly now.
      if (rel8->transport_block_to_codeword_swap_flag == 1) {
        dlsch0 = eNB->dlsch[UE_id][1];
        dlsch1 = eNB->dlsch[UE_id][0];
      }

      dlsch0_harq = dlsch0->harq_processes[rel8->harq_process];
      dlsch1_harq = dlsch1->harq_processes[rel8->harq_process];
      dlsch0->subframe_tx[subframe] = 1;
      dlsch0->harq_ids[frame%2][subframe] = rel8->harq_process;
      dlsch1->harq_ids[frame%2][subframe] = rel8->harq_process;
      //    printf("Setting DLSCH harq id %d to subframe %d\n",harq_pid,subframe);
      conv_rballoc (rel8->resource_allocation_type, rel8->resource_block_coding, fp->N_RB_DL, dlsch0_harq->rb_alloc);
      dlsch1_harq->rb_alloc[0] = dlsch0_harq->rb_alloc[0];
      dlsch0_harq->nb_rb = conv_nprb (rel8->resource_allocation_type, rel8->resource_block_coding, fp->N_RB_DL);
      dlsch1_harq->nb_rb = dlsch0_harq->nb_rb;
      AssertFatal (dlsch0_harq->nb_rb > 0, "nb_rb=0\n");
      dlsch0_harq->mcs = rel8->mcs_1;
      dlsch1_harq->mcs = rel8->mcs_2;
      dlsch0_harq->Qm = get_Qm (rel8->mcs_1);
      dlsch1_harq->Qm = get_Qm (rel8->mcs_2);
      dlsch0_harq->rvidx = rel8->redundancy_version_1;
      dlsch1_harq->rvidx = rel8->redundancy_version_2;
      // assume both TBs are active
      dlsch0_harq->Nl        = 1;
      dlsch1_harq->Nl        = 1;
#ifdef PHY_TX_THREAD
      dlsch0->active[subframe] = 1;
      dlsch1->active[subframe] = 1;
#else
      dlsch0->active = 1;
      dlsch1->active = 1;
#endif
      dlsch0->harq_mask                         |= (1<<rel8->harq_process);
      dlsch1->harq_mask                         |= (1<<rel8->harq_process);

      // check if either TB is disabled (see 36-213 V11.3 Section )
      if ((dlsch0_harq->rvidx == 1) && (dlsch0_harq->mcs == 0)) {
#ifdef PHY_TX_THREAD
        dlsch0->active[subframe] = 0;
#else
        dlsch0->active = 0;
#endif
        dlsch0->harq_mask                         &= ~(1<<rel8->harq_process);
      }

      if ((dlsch1_harq->rvidx == 1) && (dlsch1_harq->mcs == 0)) {
#ifdef PHY_TX_THREAD
        dlsch1->active[subframe]= 0;
#else
        dlsch1->active = 0;
#endif
        dlsch1->harq_mask                         &= ~(1<<rel8->harq_process);
      }

      // dlsch0_harq->dl_power_off = 0;
      // dlsch1_harq->dl_power_off = 0;

      if (fp->nb_antenna_ports_eNB == 2) {
        dlsch0_harq->TBS = TBStable[get_I_TBS (dlsch0_harq->mcs)][dlsch0_harq->nb_rb - 1];
        dlsch1_harq->TBS = TBStable[get_I_TBS (dlsch1_harq->mcs)][dlsch0_harq->nb_rb - 1];
#ifdef PHY_TX_THREAD

        if ((dlsch0->active[subframe]==1) && (dlsch1->active[subframe]==1)) {
#else

        if ((dlsch0->active==1) && (dlsch1->active==1)) {
#endif
          dlsch0_harq->mimo_mode = LARGE_CDD;
          dlsch1_harq->mimo_mode = LARGE_CDD;
          dlsch0_harq->dl_power_off = 1;
          dlsch1_harq->dl_power_off = 1;
        } else {
          dlsch0_harq->mimo_mode = ALAMOUTI;
          dlsch1_harq->mimo_mode = ALAMOUTI;
        }
      } else if (fp->nb_antenna_ports_eNB == 4) { // 4 antenna case
#ifdef PHY_TX_THREAD
        if ((dlsch0->active[subframe]==1) && (dlsch1->active[subframe]==1)) {
#else

        if ((dlsch0->active==1) && (dlsch1->active==1)) {
#endif

          switch (rel8->precoding_information) {
            case 0:                // one layer per transport block
              dlsch0_harq->mimo_mode = LARGE_CDD;
              dlsch1_harq->mimo_mode = LARGE_CDD;
              dlsch0_harq->TBS = TBStable[get_I_TBS (dlsch0_harq->mcs)][dlsch0_harq->nb_rb - 1];
              dlsch0_harq->TBS = TBStable[get_I_TBS (dlsch0_harq->mcs)][dlsch0_harq->nb_rb - 1];
              dlsch0_harq->dl_power_off = 1;
              dlsch1_harq->dl_power_off = 1;
              break;

            case 1:                // one-layers on TB 0, two on TB 1
              dlsch0_harq->mimo_mode = LARGE_CDD;
              dlsch1_harq->mimo_mode = LARGE_CDD;
              dlsch1_harq->Nl = 2;
              dlsch1_harq->TBS = TBStable[get_I_TBS (dlsch1_harq->mcs)][(dlsch1_harq->nb_rb << 1) - 1];
              dlsch0_harq->dl_power_off = 1;
              dlsch1_harq->dl_power_off = 1;
              break;

            case 2:                // two-layers on TB 0, two on TB 1
              dlsch0_harq->mimo_mode = LARGE_CDD;
              dlsch1_harq->mimo_mode = LARGE_CDD;
              dlsch0_harq->Nl = 2;
              dlsch0_harq->dl_power_off = 1;
              dlsch1_harq->dl_power_off = 1;

              if (fp->N_RB_DL <= 56) {
                dlsch0_harq->TBS = TBStable[get_I_TBS (dlsch0_harq->mcs)][(dlsch0_harq->nb_rb << 1) - 1];
                dlsch1_harq->TBS = TBStable[get_I_TBS (dlsch1_harq->mcs)][(dlsch1_harq->nb_rb << 1) - 1];
              } else {
                LOG_E (PHY, "Add implementation of Table 7.1.7.2.2-1 for two-layer TBS conversion with N_RB_DL > 56\n");
              }

              break;

            case 3:                //
              LOG_E (PHY, "Illegal value (3) for TPMI in Format 2A DCI\n");
              break;
          }

#ifdef PHY_TX_THREAD
        } else if (dlsch0->active[subframe] == 1) {
#else
        } else if (dlsch0->active == 1) {
#endif

          switch (rel8->precoding_information) {
            case 0:                // one layer per transport block
              dlsch0_harq->mimo_mode = ALAMOUTI;
              dlsch1_harq->mimo_mode = ALAMOUTI;
              dlsch0_harq->TBS = TBStable[get_I_TBS (dlsch0_harq->mcs)][dlsch0_harq->nb_rb - 1];
              break;

            case 1:                // two-layers on TB 0
              dlsch0_harq->mimo_mode = LARGE_CDD;
              dlsch0_harq->Nl = 2;
              dlsch0_harq->dl_power_off = 1;
              dlsch0_harq->TBS = TBStable[get_I_TBS (dlsch0_harq->mcs)][(dlsch0_harq->nb_rb << 1) - 1];
              break;

            case 2:                // two-layers on TB 0, two on TB 1
            case 3:                //
              LOG_E (PHY, "Illegal value %d for TPMI in Format 2A DCI with one transport block enabled\n", rel8->precoding_information);
              break;
          }

#ifdef PHY_TX_THREAD
        } else if (dlsch1->active[subframe] == 1) {
#else
        } else if (dlsch1->active == 1) {
#endif

          switch (rel8->precoding_information) {
            case 0:                // one layer per transport block
              dlsch0_harq->mimo_mode = ALAMOUTI;
              dlsch1_harq->mimo_mode = ALAMOUTI;
              dlsch1_harq->TBS = TBStable[get_I_TBS (dlsch1_harq->mcs)][dlsch1_harq->nb_rb - 1];
              break;

            case 1:                // two-layers on TB 0
              dlsch1_harq->mimo_mode = LARGE_CDD;
              dlsch1_harq->Nl = 2;
              dlsch1_harq->dl_power_off = 1;
              dlsch1_harq->TBS = TBStable[get_I_TBS (dlsch1_harq->mcs)][(dlsch1_harq->nb_rb << 1) - 1];
              break;

            case 2:                // two-layers on TB 0, two on TB 1
            case 3:                //
              LOG_E (PHY, "Illegal value %d for TPMI in Format 2A DCI with one transport block enabled\n", rel8->precoding_information);
              break;
          }
        }
      } else {
        LOG_E (PHY, "Illegal number of antennas for eNB %d\n", fp->nb_antenna_ports_eNB);
      }

      // reset HARQ process if this is the first transmission
#ifdef PHY_TX_THREAD

      if ((dlsch0->active[subframe]==1) && (dlsch0_harq->round == 0))
#else
      if ((dlsch0->active==1) && (dlsch0_harq->round == 0))
#endif
        dlsch0_harq->status = ACTIVE;

#ifdef PHY_TX_THREAD

      if ((dlsch1->active[subframe]==1) && (dlsch1_harq->round == 0))
#else
      if ((dlsch1->active==1) && (dlsch1_harq->round == 0))
#endif
        dlsch1_harq->status = ACTIVE;

      dlsch0->rnti = rel8->rnti;
      dlsch1->rnti = rel8->rnti;
      break;

    case NFAPI_DL_DCI_FORMAT_2:
      dci_alloc->format = format2;

      switch (fp->N_RB_DL) {
        case 6:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                       = sizeof_DCI2_1_5MHz_2A_TDD_t;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->mcs1     = rel8->mcs_1;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->mcs2     = rel8->mcs_2;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->ndi1     = rel8->new_data_indicator_1;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->ndi2     = rel8->new_data_indicator_2;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->rv1      = rel8->redundancy_version_1;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->rv2      = rel8->redundancy_version_2;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->TPC      = rel8->tpc;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->harq_pid = rel8->harq_process;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->tb_swap  = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->dai      = rel8->downlink_assignment_index;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->tpmi     = rel8->precoding_information;
            ((DCI2_1_5MHz_2A_TDD_t *)dci_pdu)->padding  = 0;
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                         = sizeof_DCI2_1_5MHz_2A_FDD_t;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->mcs1         = rel8->mcs_1;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->mcs2         = rel8->mcs_2;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->ndi1         = rel8->new_data_indicator_1;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->ndi2         = rel8->new_data_indicator_2;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->rballoc      = rel8->resource_block_coding;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->rv1          = rel8->redundancy_version_1;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->rv2          = rel8->redundancy_version_2;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->TPC          = rel8->tpc;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->harq_pid     = rel8->harq_process;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->tb_swap      = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->tpmi         = rel8->precoding_information;
            ((DCI2_1_5MHz_2A_FDD_t *)dci_pdu)->padding      = 0;
            //      printf("FDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          }

          break;

        case 25:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                        = sizeof_DCI2_5MHz_2A_TDD_t;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rah       = rel8->resource_allocation_type;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->mcs1      = rel8->mcs_1;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->mcs2      = rel8->mcs_2;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->ndi1      = rel8->new_data_indicator_1;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->ndi2      = rel8->new_data_indicator_2;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rballoc   = rel8->resource_block_coding;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rv1       = rel8->redundancy_version_1;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->rv2       = rel8->redundancy_version_2;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->TPC       = rel8->tpc;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->harq_pid  = rel8->harq_process;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->tb_swap   = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->dai       = rel8->downlink_assignment_index;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->tpmi      = rel8->precoding_information;
            ((DCI2_5MHz_2A_TDD_t *)dci_pdu)->padding   = 0;
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                           = sizeof_DCI2_5MHz_2A_FDD_t;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rah           = rel8->resource_allocation_type;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->mcs1          = rel8->mcs_1;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->mcs2          = rel8->mcs_2;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->ndi1          = rel8->new_data_indicator_1;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->ndi2          = rel8->new_data_indicator_2;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rballoc       = rel8->resource_block_coding;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rv1           = rel8->redundancy_version_1;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->rv2           = rel8->redundancy_version_2;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->TPC           = rel8->tpc;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->harq_pid      = rel8->harq_process;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->tb_swap       = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->tpmi          = rel8->precoding_information;
            ((DCI2_5MHz_2A_FDD_t *)dci_pdu)->padding       = 0;
            //      printf("FDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          }

          break;

        case 50:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                        = sizeof_DCI2_10MHz_2A_TDD_t;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->rah      = rel8->resource_allocation_type;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->mcs1     = rel8->mcs_1;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->mcs2     = rel8->mcs_2;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->ndi1     = rel8->new_data_indicator_1;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->ndi2     = rel8->new_data_indicator_2;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->rballoc  = rel8->resource_block_coding;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->rv1      = rel8->redundancy_version_1;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->rv2      = rel8->redundancy_version_2;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->TPC      = rel8->tpc;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->harq_pid = rel8->harq_process;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->tb_swap  = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->dai      = rel8->downlink_assignment_index;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->tpmi     = rel8->precoding_information;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->padding  = 0;
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                        = sizeof_DCI2_10MHz_2A_FDD_t;
            ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->rah       = rel8->resource_allocation_type;
            ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->mcs1      = rel8->mcs_1;
            ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->mcs2      = rel8->mcs_2;
            ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->ndi1      = rel8->new_data_indicator_1;
            ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->ndi2      = rel8->new_data_indicator_2;
            ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->rballoc   = rel8->resource_block_coding;
            ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->rv1       = rel8->redundancy_version_1;
            ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->rv2       = rel8->redundancy_version_2;
            ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->TPC       = rel8->tpc;
            ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->harq_pid  = rel8->harq_process;
            ((DCI2_10MHz_2A_FDD_t *)dci_pdu)->tb_swap   = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->tpmi      = rel8->precoding_information;
            ((DCI2_10MHz_2A_TDD_t *)dci_pdu)->padding   = 0;
          }

          break;

        case 100:
          if (fp->frame_type == TDD) {
            dci_alloc->dci_length                        = sizeof_DCI2_20MHz_2A_TDD_t;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->rah       = rel8->resource_allocation_type;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->mcs1      = rel8->mcs_1;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->mcs2      = rel8->mcs_2;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->ndi1      = rel8->new_data_indicator_1;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->ndi2      = rel8->new_data_indicator_2;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->rballoc   = rel8->resource_block_coding;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->rv1       = rel8->redundancy_version_1;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->rv2       = rel8->redundancy_version_2;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->TPC       = rel8->tpc;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->harq_pid  = rel8->harq_process;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->tb_swap   = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->dai       = rel8->downlink_assignment_index;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->tpmi      = rel8->precoding_information;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->padding   = 0;
            //        printf("TDD 1: mcs %d, rballoc %x,rv %d, NPRB %d\n",mcs,rballoc,rv,NPRB);
          } else {
            dci_alloc->dci_length                           = sizeof_DCI2_20MHz_2A_FDD_t;
            ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->rah          = rel8->resource_allocation_type;
            ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->mcs1         = rel8->mcs_1;
            ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->mcs2         = rel8->mcs_2;
            ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->ndi1         = rel8->new_data_indicator_1;
            ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->ndi2         = rel8->new_data_indicator_2;
            ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->rballoc      = rel8->resource_block_coding;
            ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->rv1          = rel8->redundancy_version_1;
            ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->rv2          = rel8->redundancy_version_2;
            ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->TPC          = rel8->tpc;
            ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->harq_pid     = rel8->harq_process;
            ((DCI2_20MHz_2A_FDD_t *)dci_pdu)->tb_swap      = rel8->transport_block_to_codeword_swap_flag;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->tpmi         = rel8->precoding_information;
            ((DCI2_20MHz_2A_TDD_t *)dci_pdu)->padding      = 0;
          }

          break;
      }

      AssertFatal (rel8->harq_process >= 8, "Format 2_2A: harq_pid=%d >= 8\n", rel8->harq_process);
      // Flip the TB to codeword mapping as described in 5.3.3.1.5 of 36-212 V11.3.0
      // note that we must set tbswap=0 in eNB scheduler if one TB is deactivated
      TB0_active = 1;
      TB1_active = 1;

      if ((rel8->redundancy_version_1 == 1) && (rel8->mcs_1 == 0)) {
        TB0_active = 0;
      }

      if ((rel8->redundancy_version_2 == 1) && (rel8->mcs_2 == 0)) {
        TB1_active = 0;
      }

#ifdef DEBUG_HARQ
      printf ("RV0 = %d, RV1 = %d. MCS0 = %d, MCS1=%d\n", rel8->redundancy_version_1, rel8->redundancy_version_2, rel8->mcs_1, rel8->mcs_2);
#endif

      if (TB0_active && TB1_active && rel8->transport_block_to_codeword_swap_flag==0) {
#ifdef PHY_TX_THREAD
        dlsch0->active[subframe] = 1;
        dlsch1->active[subframe] = 1;
#else
        dlsch0->active = 1;
        dlsch1->active = 1;
#endif
        dlsch0->harq_mask                         |= (1<<rel8->harq_process);
        dlsch1->harq_mask                         |= (1<<rel8->harq_process);
        dlsch0_harq = dlsch0->harq_processes[rel8->harq_process];
        dlsch1_harq = dlsch1->harq_processes[rel8->harq_process];
        dlsch0_harq->mcs = rel8->mcs_1;
        dlsch1_harq->mcs = rel8->mcs_2;
        dlsch0_harq->Qm = get_Qm (rel8->mcs_1);
        dlsch1_harq->Qm = get_Qm (rel8->mcs_2);
        dlsch0_harq->rvidx = rel8->redundancy_version_1;
        dlsch1_harq->rvidx = rel8->redundancy_version_2;
        dlsch0_harq->status = ACTIVE;
        dlsch1_harq->status = ACTIVE;
        dlsch0_harq->codeword = 0;
        dlsch1_harq->codeword = 1;
#ifdef DEBUG_HARQ
        printf ("\n ENB: BOTH ACTIVE\n");
#endif
      } else if (TB0_active && TB1_active && rel8->transport_block_to_codeword_swap_flag == 1) {
        dlsch0 = eNB->dlsch[UE_id][1];
        dlsch1 = eNB->dlsch[UE_id][0];
#ifdef PHY_TX_THREAD
        dlsch0->active[subframe] = 1;
        dlsch1->active[subframe] = 1;
#else
        dlsch0->active = 1;
        dlsch1->active = 1;
#endif
        dlsch0->harq_mask |= (1 << rel8->harq_process);
        dlsch1->harq_mask |= (1 << rel8->harq_process);
        dlsch1_harq = dlsch1->harq_processes[rel8->harq_process];
        dlsch0_harq->mcs = rel8->mcs_1;
        dlsch1_harq->mcs = rel8->mcs_2;
        dlsch0_harq->Qm = get_Qm (rel8->mcs_1);
        dlsch1_harq->Qm = get_Qm (rel8->mcs_2);
        dlsch0_harq->rvidx = rel8->redundancy_version_1;
        dlsch1_harq->rvidx = rel8->redundancy_version_2;
        dlsch0_harq->status = ACTIVE;
        dlsch1_harq->status = ACTIVE;
        dlsch0_harq->codeword=1;
        dlsch1_harq->codeword=0;
      } else if (TB0_active && (TB1_active==0)) {
#ifdef PHY_TX_THREAD
        dlsch0->active[subframe] = 1;
#else
        dlsch0->active = 1;
#endif
        dlsch0->harq_mask                         |= (1<<rel8->harq_process);
        dlsch0_harq = dlsch0->harq_processes[rel8->harq_process];
        dlsch0_harq->mcs = rel8->mcs_1;
        dlsch0_harq->Qm = get_Qm (rel8->mcs_1);
        dlsch0_harq->rvidx = rel8->redundancy_version_1;
        dlsch0_harq->status = ACTIVE;
        dlsch0_harq->codeword = 0;
        dlsch1 = NULL;
        dlsch1_harq = NULL;
#ifdef DEBUG_HARQ
        printf ("\n ENB: TB1 is deactivated, retransmit TB0 transmit in TM6\n");
#endif
      } else if ((TB0_active==0) && TB1_active) {
#ifdef PHY_TX_THREAD
        dlsch1->active[subframe] = 1;
#else
        dlsch1->active = 1;
#endif
        dlsch1->harq_mask                         |= (1<<rel8->harq_process);
        dlsch1_harq = dlsch1->harq_processes[rel8->harq_process];
        dlsch1_harq->mcs = rel8->mcs_2;
        dlsch1_harq->Qm = get_Qm (rel8->mcs_2);
        dlsch1_harq->rvidx = rel8->redundancy_version_2;
        dlsch1_harq->status = ACTIVE;
        dlsch1_harq->codeword = 0;
        dlsch0 = NULL;
        dlsch0_harq = NULL;
#ifdef DEBUG_HARQ
        printf ("\n ENB: TB0 is deactivated, retransmit TB1 transmit in TM6\n");
#endif
      }

      if (dlsch0 != NULL) {
        dlsch0->subframe_tx[subframe] = 1;
        dlsch0->harq_ids[frame%2][subframe] = rel8->harq_process;
      }

      if (dlsch1_harq != NULL) {
        dlsch1->harq_ids[frame%2][subframe] = rel8->harq_process;
      }

      if (dlsch0 != NULL) {
        conv_rballoc (rel8->resource_allocation_type, rel8->resource_block_coding, fp->N_RB_DL, dlsch0_harq->rb_alloc);
        dlsch0_harq->nb_rb = conv_nprb (rel8->resource_allocation_type, rel8->resource_block_coding, fp->N_RB_DL);

        if (dlsch1 != NULL) {
          dlsch1_harq->rb_alloc[0] = dlsch0_harq->rb_alloc[0];
          dlsch1_harq->nb_rb = dlsch0_harq->nb_rb;
        }
      } else if ((dlsch0 == NULL) && (dlsch1 != NULL)) {
        conv_rballoc (rel8->resource_allocation_type, rel8->resource_block_coding, fp->N_RB_DL, dlsch1_harq->rb_alloc);
        dlsch1_harq->nb_rb = conv_nprb (rel8->resource_allocation_type, rel8->resource_block_coding, fp->N_RB_DL);
      }

      // assume both TBs are active
      if (dlsch0_harq != NULL)
        dlsch0_harq->Nl = 1;

      if (dlsch1_harq != NULL)
        dlsch1_harq->Nl = 1;

      // check if either TB is disabled (see 36-213 V11.3 Section )

      if (fp->nb_antenna_ports_eNB == 2) {
        if ((dlsch0 != NULL) && (dlsch1 != NULL)) {       //two CW active
          dlsch0_harq->dl_power_off = 1;
          dlsch1_harq->dl_power_off = 1;
          dlsch0_harq->TBS = TBStable[get_I_TBS (dlsch0_harq->mcs)][dlsch0_harq->nb_rb - 1];
          dlsch1_harq->TBS = TBStable[get_I_TBS (dlsch1_harq->mcs)][dlsch1_harq->nb_rb - 1];

          switch (rel8->precoding_information) {
            case 0:
              dlsch0_harq->mimo_mode = DUALSTREAM_UNIFORM_PRECODING1;
              dlsch1_harq->mimo_mode = DUALSTREAM_UNIFORM_PRECODING1;
              dlsch0_harq->pmi_alloc = pmi_extend (fp, 0, 1);
              dlsch1_harq->pmi_alloc = pmi_extend (fp, 0, 1);
              break;

            case 1:
              dlsch0_harq->mimo_mode = DUALSTREAM_UNIFORM_PRECODINGj;
              dlsch1_harq->mimo_mode = DUALSTREAM_UNIFORM_PRECODINGj;
              dlsch0_harq->pmi_alloc = pmi_extend (fp, 1, 1);
              dlsch0_harq->pmi_alloc = pmi_extend (fp, 1, 1);
              break;

            case 2:                // PUSCH precoding
              dlsch0_harq->mimo_mode = DUALSTREAM_PUSCH_PRECODING;
              dlsch0_harq->pmi_alloc = DL_pmi_single;
              dlsch1_harq->mimo_mode = DUALSTREAM_PUSCH_PRECODING;
              dlsch1_harq->pmi_alloc = DL_pmi_single;
              break;

            default:
              break;
          }
        } else if ((dlsch0 != NULL) && (dlsch1 == NULL)) {        // only CW 0 active
          dlsch0_harq->dl_power_off = 1;
          dlsch0_harq->TBS = TBStable[get_I_TBS (dlsch0_harq->mcs)][dlsch0_harq->nb_rb - 1];

          switch (rel8->precoding_information) {
            case 0:
              dlsch0_harq->mimo_mode = ALAMOUTI;
              break;

            case 1:
              dlsch0_harq->mimo_mode = UNIFORM_PRECODING11;
              dlsch0_harq->pmi_alloc = pmi_extend (fp, 0, 0);
              break;

            case 2:
              dlsch0_harq->mimo_mode = UNIFORM_PRECODING1m1;
              dlsch0_harq->pmi_alloc = pmi_extend (fp, 1, 0);
              break;

            case 3:
              dlsch0_harq->mimo_mode = UNIFORM_PRECODING1j;
              dlsch0_harq->pmi_alloc = pmi_extend (fp, 2, 0);
              break;

            case 4:
              dlsch0_harq->mimo_mode = UNIFORM_PRECODING1mj;
              dlsch0_harq->pmi_alloc = pmi_extend (fp, 3, 0);
              break;

            case 5:
              dlsch0_harq->mimo_mode = PUSCH_PRECODING0;
              dlsch0_harq->pmi_alloc = DL_pmi_single;
              break;

            case 6:
              dlsch0_harq->mimo_mode = PUSCH_PRECODING1;
              dlsch0_harq->pmi_alloc = DL_pmi_single;
              break;
          }
        } else if ((dlsch0 == NULL) && (dlsch1 != NULL)) {
          dlsch1_harq->dl_power_off = 1;
          dlsch1_harq->TBS = TBStable[get_I_TBS (dlsch1_harq->mcs)][dlsch1_harq->nb_rb - 1];

          switch (rel8->precoding_information) {
            case 0:
              dlsch1_harq->mimo_mode = ALAMOUTI;
              break;

            case 1:
              dlsch1_harq->mimo_mode = UNIFORM_PRECODING11;
              dlsch1_harq->pmi_alloc = pmi_extend (fp, 0, 0);
              break;

            case 2:
              dlsch1_harq->mimo_mode = UNIFORM_PRECODING1m1;
              dlsch1_harq->pmi_alloc = pmi_extend (fp, 1, 0);
              break;

            case 3:
              dlsch1_harq->mimo_mode = UNIFORM_PRECODING1j;
              dlsch1_harq->pmi_alloc = pmi_extend (fp, 2, 0);
              break;

            case 4:
              dlsch1_harq->mimo_mode = UNIFORM_PRECODING1mj;
              dlsch1_harq->pmi_alloc = pmi_extend (fp, 3, 0);
              break;

            case 5:
              dlsch1_harq->mimo_mode = PUSCH_PRECODING0;
              dlsch1_harq->pmi_alloc = DL_pmi_single;
              break;

            case 6:
              dlsch1_harq->mimo_mode = PUSCH_PRECODING1;
              dlsch1_harq->pmi_alloc = DL_pmi_single;
              break;
          }
        }
      } else if (fp->nb_antenna_ports_eNB == 4) {
        // fill in later
      }

      // reset HARQ process if this is the first transmission
      /* if (dlsch0_harq->round == 0)
         dlsch0_harq->status = ACTIVE;

         if (dlsch1_harq->round == 0)
         dlsch1_harq->status = ACTIVE; */
      if (dlsch0_harq != NULL)
        dlsch0->rnti = rel8->rnti;

      if (dlsch1 != NULL)
        dlsch1->rnti = rel8->rnti;

      break;
  }

  if (dlsch0_harq) {
    dlsch0_harq->frame    = frame;
    dlsch0_harq->subframe = subframe;
  }

  if (dlsch1_harq) {
    dlsch1_harq->frame    = frame;
    dlsch1_harq->subframe = subframe;
  }

#ifdef DEBUG_DCI

  if (dlsch0) {
    printf ("dlsch0 eNB: dlsch0   %p\n", dlsch0);
    printf ("dlsch0 eNB: rnti     %x\n", dlsch0->rnti);
    printf ("dlsch0 eNB: NBRB     %d\n", dlsch0_harq->nb_rb);
    printf ("dlsch0 eNB: rballoc  %x\n", dlsch0_harq->rb_alloc[0]);
    printf ("dlsch0 eNB: harq_pid %d\n", harq_pid);
    printf ("dlsch0 eNB: round    %d\n", dlsch0_harq->round);
    printf ("dlsch0 eNB: rvidx    %d\n", dlsch0_harq->rvidx);
    printf ("dlsch0 eNB: TBS      %d (NPRB %d)\n", dlsch0_harq->TBS, NPRB);
    printf ("dlsch0 eNB: mcs      %d\n", dlsch0_harq->mcs);
    printf ("dlsch0 eNB: tpmi %d\n", rel8->precoding_information);
    printf ("dlsch0 eNB: mimo_mode %d\n", dlsch0_harq->mimo_mode);
  }

  if (dlsch1) {
    printf ("dlsch1 eNB: dlsch1   %p\n", dlsch1);
    printf ("dlsch1 eNB: rnti     %x\n", dlsch1->rnti);
    printf ("dlsch1 eNB: NBRB     %d\n", dlsch1_harq->nb_rb);
    printf ("dlsch1 eNB: rballoc  %x\n", dlsch1_harq->rb_alloc[0]);
    printf ("dlsch1 eNB: harq_pid %d\n", harq_pid);
    printf ("dlsch1 eNB: round    %d\n", dlsch1_harq->round);
    printf ("dlsch1 eNB: rvidx    %d\n", dlsch1_harq->rvidx);
    printf ("dlsch1 eNB: TBS      %d (NPRB %d)\n", dlsch1_harq->TBS, NPRB);
    printf ("dlsch1 eNB: mcs      %d\n", dlsch1_harq->mcs);
    printf ("dlsch1 eNB: tpmi %d\n", rel8->precoding_information);
    printf ("dlsch1 eNB: mimo_mode %d\n", dlsch1_harq->mimo_mode);
  }

#endif
  //printf("DCI %d.%d rnti %d harq %d TBS %d\n", frame, subframe, rel8->rnti, rel8->harq_process, dlsch0_harq->TBS);
#if T_TRACER

  if (dlsch0->active)
    T(T_ENB_PHY_DLSCH_UE_DCI, T_INT(0), T_INT(frame), T_INT(subframe),
      T_INT(rel8->rnti), T_INT(rel8->dci_format), T_INT(rel8->harq_process),
      T_INT(rel8->mcs_1), T_INT(dlsch0_harq->TBS));

#endif
}


void fill_mdci_and_dlsch(PHY_VARS_eNB *eNB,L1_rxtx_proc_t *proc,mDCI_ALLOC_t *dci_alloc,nfapi_dl_config_mpdcch_pdu *pdu) {
  LTE_DL_FRAME_PARMS *fp = &eNB->frame_parms;
  uint8_t        *dci_pdu = &dci_alloc->dci_pdu[0];
  nfapi_dl_config_mpdcch_pdu_rel13_t *rel13 = &pdu->mpdcch_pdu_rel13;
  LTE_eNB_DLSCH_t *dlsch0 = NULL;
  LTE_DL_eNB_HARQ_t *dlsch0_harq = NULL;
  int             UE_id;
  int             subframe = proc->subframe_tx;
  int             frame = proc->frame_tx;
  dci_alloc->firstCCE = rel13->ecce_index;
  dci_alloc->L = rel13->aggregation_level;
  dci_alloc->rnti = rel13->rnti;
  dci_alloc->harq_pid = rel13->harq_process;
  dci_alloc->narrowband = rel13->mpdcch_narrow_band;
  dci_alloc->number_of_prb_pairs = rel13->number_of_prb_pairs;
  dci_alloc->resource_block_assignment = rel13->resource_block_assignment;
  dci_alloc->transmission_type = rel13->mpdcch_tansmission_type;
  dci_alloc->start_symbol = rel13->start_symbol;
  dci_alloc->ce_mode = rel13->ce_mode;
  dci_alloc->dmrs_scrambling_init = rel13->drms_scrambling_init;
  dci_alloc->i0 = rel13->initial_transmission_sf_io;
  UE_id = find_dlsch (rel13->rnti, eNB, SEARCH_EXIST_OR_FREE);
  AssertFatal (UE_id != -1, "no free or exiting dlsch_context\n");
  AssertFatal (UE_id < NUMBER_OF_UE_MAX, "returned UE_id %d >= %d(NUMBER_OF_UE_MAX)\n", UE_id, NUMBER_OF_UE_MAX);
  dlsch0 = eNB->dlsch[UE_id][0];
  dlsch0_harq = dlsch0->harq_processes[rel13->harq_process];
  dci_alloc->ra_flag = 0;

  if (rel13->rnti_type == 2) {
    dci_alloc->ra_flag = 1;
  }

  AssertFatal (fp->frame_type == FDD, "TDD is not supported yet for eMTC\n");
  AssertFatal (fp->N_RB_DL == 25 || fp->N_RB_DL == 50 || fp->N_RB_DL == 100, "eMTC only with N_RB_DL = 25,50,100\n");

  switch (rel13->dci_format) {
    case 10:                     // Format 6-1A
      dci_alloc->format = format6_1A;

      switch (fp->N_RB_DL) {
        case 25:
          dci_alloc->dci_length = sizeof_DCI6_1A_5MHz_t;
          ((DCI6_1A_5MHz_t *) dci_pdu)->type = 1;
          ((DCI6_1A_5MHz_t *) dci_pdu)->hopping = rel13->frequency_hopping_enabled_flag;
          ((DCI6_1A_5MHz_t *) dci_pdu)->rballoc = rel13->resource_block_coding;
          ((DCI6_1A_5MHz_t *) dci_pdu)->mcs = rel13->mcs;
          ((DCI6_1A_5MHz_t *) dci_pdu)->rep = rel13->pdsch_reptition_levels;
          ((DCI6_1A_5MHz_t *) dci_pdu)->harq_pid = rel13->harq_process;
          ((DCI6_1A_5MHz_t *) dci_pdu)->ndi = rel13->new_data_indicator;
          ((DCI6_1A_5MHz_t *) dci_pdu)->rv = rel13->redundancy_version;
          ((DCI6_1A_5MHz_t *) dci_pdu)->TPC = rel13->tpc;
          ((DCI6_1A_5MHz_t *) dci_pdu)->srs_req = rel13->srs_request;
          ((DCI6_1A_5MHz_t *) dci_pdu)->harq_ack_off = rel13->harq_resource_offset;
          ((DCI6_1A_5MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
          break;

        case 50:
          dci_alloc->dci_length = sizeof_DCI6_1A_10MHz_t;
          ((DCI6_1A_10MHz_t *) dci_pdu)->type = 1;
          ((DCI6_1A_10MHz_t *) dci_pdu)->hopping = rel13->frequency_hopping_enabled_flag;
          ((DCI6_1A_10MHz_t *) dci_pdu)->rballoc = rel13->resource_block_coding&31;
          ((DCI6_1A_10MHz_t *) dci_pdu)->narrowband = rel13->resource_block_coding>>5;
          ((DCI6_1A_10MHz_t *) dci_pdu)->mcs = rel13->mcs;
          ((DCI6_1A_10MHz_t *) dci_pdu)->rep = (rel13->pdsch_reptition_levels);
          ((DCI6_1A_10MHz_t *) dci_pdu)->harq_pid = rel13->harq_process;
          ((DCI6_1A_10MHz_t *) dci_pdu)->ndi = rel13->new_data_indicator;
          ((DCI6_1A_10MHz_t *) dci_pdu)->rv = rel13->redundancy_version;
          ((DCI6_1A_10MHz_t *) dci_pdu)->TPC = rel13->tpc;
          ((DCI6_1A_10MHz_t *) dci_pdu)->srs_req = rel13->srs_request;
          ((DCI6_1A_10MHz_t *) dci_pdu)->harq_ack_off = rel13->harq_resource_offset;
          ((DCI6_1A_10MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
          LOG_D(PHY,
                "Frame %d, Subframe %d : Programming Format 6-1A DCI, type %d, hopping %d, narrowband %d, rballoc %x, mcs %d, rep %d, harq_pid %d, ndi %d, rv %d, TPC %d, srs_req %d, harq_ack_off %d, dci_rep r%d => %x\n",
                frame,subframe,
                ((DCI6_1A_10MHz_t *) dci_pdu)->type,
                ((DCI6_1A_10MHz_t *) dci_pdu)->hopping,
                ((DCI6_1A_10MHz_t *) dci_pdu)->narrowband,
                ((DCI6_1A_10MHz_t *) dci_pdu)->rballoc,
                ((DCI6_1A_10MHz_t *) dci_pdu)->mcs,
                ((DCI6_1A_10MHz_t *) dci_pdu)->rep,
                ((DCI6_1A_10MHz_t *) dci_pdu)->harq_pid,
                ((DCI6_1A_10MHz_t *) dci_pdu)->ndi,
                ((DCI6_1A_10MHz_t *) dci_pdu)->rv,
                ((DCI6_1A_10MHz_t *) dci_pdu)->TPC,
                ((DCI6_1A_10MHz_t *) dci_pdu)->srs_req,
                ((DCI6_1A_10MHz_t *) dci_pdu)->harq_ack_off,
                ((DCI6_1A_10MHz_t *) dci_pdu)->dci_rep,
                ((uint32_t *)dci_pdu)[0]);
          break;

        case 100:
          dci_alloc->dci_length = sizeof_DCI6_1A_20MHz_t;
          ((DCI6_1A_20MHz_t *) dci_pdu)->type = 1;
          ((DCI6_1A_20MHz_t *) dci_pdu)->hopping = rel13->frequency_hopping_enabled_flag;
          ((DCI6_1A_20MHz_t *) dci_pdu)->rballoc = rel13->resource_block_coding;
          ((DCI6_1A_20MHz_t *) dci_pdu)->mcs = rel13->mcs;
          ((DCI6_1A_20MHz_t *) dci_pdu)->rep = rel13->pdsch_reptition_levels;
          ((DCI6_1A_20MHz_t *) dci_pdu)->harq_pid = rel13->harq_process;
          ((DCI6_1A_20MHz_t *) dci_pdu)->ndi = rel13->new_data_indicator;
          ((DCI6_1A_20MHz_t *) dci_pdu)->rv = rel13->redundancy_version;
          ((DCI6_1A_20MHz_t *) dci_pdu)->TPC = rel13->tpc;
          ((DCI6_1A_20MHz_t *) dci_pdu)->srs_req = rel13->srs_request;
          ((DCI6_1A_20MHz_t *) dci_pdu)->harq_ack_off = rel13->harq_resource_offset;
          ((DCI6_1A_20MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
          break;
      }

      break;

    case 11:                     // Format 6-1B
      dci_alloc->format = format6_1B;

      switch (fp->N_RB_DL) {
        case 25:
          dci_alloc->dci_length = sizeof_DCI6_1B_5MHz_t;
          ((DCI6_1B_5MHz_t *) dci_pdu)->type = 1;
          ((DCI6_1B_5MHz_t *) dci_pdu)->rballoc = rel13->resource_block_coding;
          ((DCI6_1B_5MHz_t *) dci_pdu)->mcs = rel13->mcs;
          ((DCI6_1B_5MHz_t *) dci_pdu)->rep = rel13->pdsch_reptition_levels;
          ((DCI6_1B_5MHz_t *) dci_pdu)->harq_pid = rel13->harq_process;
          ((DCI6_1B_5MHz_t *) dci_pdu)->ndi = rel13->new_data_indicator;
          ((DCI6_1B_5MHz_t *) dci_pdu)->harq_ack_off = rel13->harq_resource_offset;
          ((DCI6_1B_5MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
          break;

        case 50:
          dci_alloc->dci_length = sizeof_DCI6_1B_10MHz_t;
          ((DCI6_1B_10MHz_t *) dci_pdu)->type = 1;
          ((DCI6_1B_10MHz_t *) dci_pdu)->rballoc = rel13->resource_block_coding;
          ((DCI6_1B_10MHz_t *) dci_pdu)->mcs = rel13->mcs;
          ((DCI6_1B_10MHz_t *) dci_pdu)->rep = rel13->pdsch_reptition_levels;
          ((DCI6_1B_10MHz_t *) dci_pdu)->harq_pid = rel13->harq_process;
          ((DCI6_1B_10MHz_t *) dci_pdu)->ndi = rel13->new_data_indicator;
          ((DCI6_1B_10MHz_t *) dci_pdu)->harq_ack_off = rel13->harq_resource_offset;
          ((DCI6_1B_10MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
          break;

        case 100:
          dci_alloc->dci_length = sizeof_DCI6_1B_20MHz_t;
          ((DCI6_1B_20MHz_t *) dci_pdu)->type = 1;
          ((DCI6_1B_20MHz_t *) dci_pdu)->rballoc = rel13->resource_block_coding;
          ((DCI6_1B_20MHz_t *) dci_pdu)->mcs = rel13->mcs;
          ((DCI6_1B_20MHz_t *) dci_pdu)->rep = rel13->pdsch_reptition_levels;
          ((DCI6_1B_20MHz_t *) dci_pdu)->harq_pid = rel13->harq_process;
          ((DCI6_1B_20MHz_t *) dci_pdu)->ndi = rel13->new_data_indicator;
          ((DCI6_1B_20MHz_t *) dci_pdu)->harq_ack_off = rel13->harq_resource_offset;
          ((DCI6_1B_20MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
          break;
      }

    case 12:                     // Format 6-2
      dci_alloc->format = format6_2;

      switch (fp->N_RB_DL) {
        case 25:
          dci_alloc->dci_length = sizeof_DCI6_2_5MHz_t;

          if (rel13->paging_direct_indication_differentiation_flag == 0) {
            ((DCI6_2_di_5MHz_t *) dci_pdu)->type = 0;
            ((DCI6_2_di_5MHz_t *) dci_pdu)->di_info = rel13->direct_indication;
          } else {
            ((DCI6_2_paging_5MHz_t *) dci_pdu)->type = 1;
            ((DCI6_2_paging_5MHz_t *) dci_pdu)->rballoc = rel13->resource_block_coding;
            ((DCI6_2_paging_5MHz_t *) dci_pdu)->mcs = rel13->mcs;
            ((DCI6_2_paging_5MHz_t *) dci_pdu)->rep = rel13->pdsch_reptition_levels;
            ((DCI6_2_paging_5MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
          }

          break;

        case 50:
          dci_alloc->dci_length = sizeof_DCI6_2_10MHz_t;

          if (rel13->paging_direct_indication_differentiation_flag == 0) {
            ((DCI6_2_di_10MHz_t *) dci_pdu)->type = 0;
            ((DCI6_2_di_10MHz_t *) dci_pdu)->di_info = rel13->direct_indication;
          } else {
            ((DCI6_2_paging_10MHz_t *) dci_pdu)->type = 1;
            ((DCI6_2_paging_10MHz_t *) dci_pdu)->rballoc = rel13->resource_block_coding;
            ((DCI6_2_paging_10MHz_t *) dci_pdu)->mcs = rel13->mcs;
            ((DCI6_2_paging_10MHz_t *) dci_pdu)->rep = rel13->pdsch_reptition_levels;
            ((DCI6_2_paging_10MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
          }

          break;

        case 100:
          dci_alloc->dci_length = sizeof_DCI6_2_20MHz_t;

          if (rel13->paging_direct_indication_differentiation_flag == 0) {
            ((DCI6_2_di_20MHz_t *) dci_pdu)->type = 0;
            ((DCI6_2_di_20MHz_t *) dci_pdu)->di_info = rel13->direct_indication;
          } else {
            ((DCI6_2_paging_20MHz_t *) dci_pdu)->type = 1;
            ((DCI6_2_paging_20MHz_t *) dci_pdu)->rballoc = rel13->resource_block_coding;
            ((DCI6_2_paging_20MHz_t *) dci_pdu)->mcs = rel13->mcs;
            ((DCI6_2_paging_20MHz_t *) dci_pdu)->rep = rel13->pdsch_reptition_levels;
            ((DCI6_2_paging_20MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
          }

          break;
      }
  }

  AssertFatal (rel13->harq_process < 8, "ERROR: Format 6_1A: harq_pid=%d >= 8\n", rel13->harq_process);
  dlsch0->ue_type = rel13->ce_mode;
  dlsch0_harq = dlsch0->harq_processes[rel13->harq_process];
  dlsch0_harq->codeword = 0;
  // printf("DCI: Setting subframe_tx for subframe %d\n",subframe);
  dlsch0->subframe_tx[(subframe + 2) % 10] = 1;
  LOG_D(PHY,"PDSCH : resource_block_coding %x\n",rel13->resource_block_coding);
  conv_eMTC_rballoc (rel13->resource_block_coding,
                     fp->N_RB_DL,
                     dlsch0_harq->rb_alloc);
  dlsch0_harq->nb_rb = RIV2nb_rb_LUT6[rel13->resource_block_coding & 31];       // this is the 6PRB RIV
  dlsch0_harq->rvidx = rel13->redundancy_version;
  dlsch0_harq->Nl = 1;
  //    dlsch[0]->layer_index = 0;
  //  if (beamforming_mode == 0)
  dlsch0_harq->mimo_mode = (fp->nb_antenna_ports_eNB == 1) ? SISO : ALAMOUTI;
  //else if (beamforming_mode == 7)
  //  dlsch0_harq->mimo_mode = TM7;
  //else
  //LOG_E(PHY,"Invalid beamforming mode %dL\n", beamforming_mode);
  dlsch0_harq->dl_power_off = 1;
  dlsch0->subframe_tx[subframe] = 1;

  if (dlsch0->rnti != rel13->rnti) {     // if rnti of dlsch is not the same as in the config, this is a new entry
    dlsch0_harq->round = 0;
    dlsch0->harq_mask =0;
    printf("*********************** rnti %x => %x, pos %d\n",rel13->rnti,dlsch0->rnti,UE_id);
  }

  if ((dlsch0->harq_mask & (1 << rel13->harq_process)) > 0) {
    if ((rel13->new_data_indicator != dlsch0_harq->ndi)||(dci_alloc->ra_flag==1))
      dlsch0_harq->round = 0;
  } else {                      // process is inactive, so activate and set round to 0
    dlsch0_harq->round = 0;
  }

  dlsch0_harq->ndi = rel13->new_data_indicator;

  if (dlsch0_harq->round == 0) {
    dlsch0_harq->status = ACTIVE;
    dlsch0_harq->mcs = rel13->mcs;

    if (dci_alloc->ra_flag == 0) // get TBS from table using mcs and nb_rb
      dlsch0_harq->TBS         = TBStable[get_I_TBS(dlsch0_harq->mcs)][dlsch0_harq->nb_rb-1];
    else if (rel13->tpc == 0)  //N1A_PRB=2, get TBS from table using mcs and nb_rb=2
      dlsch0_harq->TBS         = TBStable[get_I_TBS(dlsch0_harq->mcs)][1];
    else if (rel13->tpc == 1)  //N1A_PRB=3, get TBS from table using mcs and nb_rb=3
      dlsch0_harq->TBS         = TBStable[get_I_TBS(dlsch0_harq->mcs)][2];
    else AssertFatal(1==0,"Don't know how to set TBS (TPC %d)\n",rel13->tpc);

    LOG_D(PHY,"fill_mdci_and_dlsch : TBS = %d(%d) %p, %x\n",dlsch0_harq->TBS,dlsch0_harq->mcs,dlsch0,rel13->rnti);
  }

  dlsch0->active = 1;
  dlsch0->harq_mask |= (1 << rel13->harq_process);
  dlsch0_harq->frame    = (subframe >= 8) ? ((frame + 1) & 1023) : frame;
  dlsch0_harq->subframe = (subframe + 2) % 10;
  LOG_D(PHY,"Setting DLSCH UEid %d harq_ids[%d] from %d to %d\n",UE_id,dlsch0_harq->subframe,dlsch0->harq_ids[frame%2][dlsch0_harq->subframe],rel13->harq_process);
  dlsch0->harq_ids[dlsch0_harq->frame%2][dlsch0_harq->subframe] = rel13->harq_process;
  dlsch0_harq->pdsch_start = rel13->start_symbol;
  LOG_D(PHY,"Setting DLSCH harq %d round %d to active for %d.%d\n",rel13->harq_process,dlsch0_harq->round,dlsch0_harq->frame,dlsch0_harq->subframe);
  dlsch0->rnti = rel13->rnti;
  dlsch0_harq->Qm = get_Qm(rel13->mcs);
}

void fill_dci0(PHY_VARS_eNB *eNB,int frame,int subframe,L1_rxtx_proc_t *proc,
               DCI_ALLOC_t *dci_alloc,nfapi_hi_dci0_dci_pdu *pdu) {
  LTE_DL_FRAME_PARMS *frame_parms = &eNB->frame_parms;
  uint32_t        cqi_req = pdu->dci_pdu_rel8.cqi_csi_request;
  uint32_t        dai = pdu->dci_pdu_rel8.dl_assignment_index;
  uint32_t        cshift = pdu->dci_pdu_rel8.cyclic_shift_2_for_drms;
  uint32_t        TPC = pdu->dci_pdu_rel8.tpc;
  uint32_t        mcs = pdu->dci_pdu_rel8.mcs_1;
  uint32_t        hopping = pdu->dci_pdu_rel8.frequency_hopping_enabled_flag;
  uint32_t        rballoc = computeRIV (frame_parms->N_RB_DL,
                                        pdu->dci_pdu_rel8.resource_block_start,
                                        pdu->dci_pdu_rel8.number_of_resource_block);
  uint32_t        ndi = pdu->dci_pdu_rel8.new_data_indication_1;
  uint16_t UE_id   = -1;
#ifdef T_TRACER
  T(T_ENB_PHY_ULSCH_UE_DCI, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe),
    T_INT(pdu->dci_pdu_rel8.rnti), T_INT(pdu->dci_pdu_rel8.harq_pid),
    T_INT(mcs), T_INT(-1 /* TODO: remove round? */),
    T_INT(pdu->dci_pdu_rel8.resource_block_start),
    T_INT(pdu->dci_pdu_rel8.number_of_resource_block),
    T_INT(get_TBS_UL(mcs, pdu->dci_pdu_rel8.number_of_resource_block) * 8),
    T_INT(pdu->dci_pdu_rel8.aggregation_level),
    T_INT(pdu->dci_pdu_rel8.cce_index));
#endif
  void           *dci_pdu = (void *) dci_alloc->dci_pdu;
  LOG_D(PHY,"SFN/SF:%04d%d DCI0[rnti %x cqi %d mcs %d hopping %d rballoc %x (%d,%d) ndi %d TPC %d cshift %d]\n",
        frame,subframe,
        pdu->dci_pdu_rel8.rnti,cqi_req, mcs,hopping,rballoc,
        pdu->dci_pdu_rel8.resource_block_start,
        pdu->dci_pdu_rel8.number_of_resource_block, ndi,TPC,cshift);
  dci_alloc->format = format0;
  dci_alloc->firstCCE = pdu->dci_pdu_rel8.cce_index;
  dci_alloc->L = pdu->dci_pdu_rel8.aggregation_level;
  dci_alloc->rnti = pdu->dci_pdu_rel8.rnti;
  dci_alloc->ra_flag = 0;

  switch (frame_parms->N_RB_DL) {
    case 6:
      if (frame_parms->frame_type == TDD) {
        ((DCI0_1_5MHz_TDD_1_6_t *) dci_pdu)->cqi_req = cqi_req;
        ((DCI0_1_5MHz_TDD_1_6_t *) dci_pdu)->dai = dai;
        ((DCI0_1_5MHz_TDD_1_6_t *) dci_pdu)->cshift = cshift;
        ((DCI0_1_5MHz_TDD_1_6_t *) dci_pdu)->TPC = TPC;
        ((DCI0_1_5MHz_TDD_1_6_t *) dci_pdu)->mcs = mcs;
        ((DCI0_1_5MHz_TDD_1_6_t *) dci_pdu)->ndi = ndi;
        ((DCI0_1_5MHz_TDD_1_6_t *) dci_pdu)->rballoc = rballoc;
        ((DCI0_1_5MHz_TDD_1_6_t *) dci_pdu)->hopping = hopping;
        ((DCI0_1_5MHz_TDD_1_6_t *) dci_pdu)->type = 0;
        ((DCI0_1_5MHz_TDD_1_6_t *) dci_pdu)->padding = 0;
        dci_alloc->dci_length = sizeof_DCI0_1_5MHz_TDD_1_6_t;
      } else {
        ((DCI0_1_5MHz_FDD_t *) dci_pdu)->cqi_req = cqi_req;
        ((DCI0_1_5MHz_FDD_t *) dci_pdu)->cshift = cshift;
        ((DCI0_1_5MHz_FDD_t *) dci_pdu)->TPC = TPC;
        ((DCI0_1_5MHz_FDD_t *) dci_pdu)->mcs = mcs;
        ((DCI0_1_5MHz_FDD_t *) dci_pdu)->ndi = ndi;
        ((DCI0_1_5MHz_FDD_t *) dci_pdu)->rballoc = rballoc;
        ((DCI0_1_5MHz_FDD_t *) dci_pdu)->hopping = hopping;
        ((DCI0_1_5MHz_FDD_t *) dci_pdu)->type = 0;
        ((DCI0_1_5MHz_FDD_t *) dci_pdu)->padding = 0;
        dci_alloc->dci_length = sizeof_DCI0_1_5MHz_FDD_t;
      }

      break;

    case 25:
      if (frame_parms->frame_type == TDD) {
        ((DCI0_5MHz_TDD_1_6_t *) dci_pdu)->cqi_req = cqi_req;
        ((DCI0_5MHz_TDD_1_6_t *) dci_pdu)->dai = dai;
        ((DCI0_5MHz_TDD_1_6_t *) dci_pdu)->cshift = cshift;
        ((DCI0_5MHz_TDD_1_6_t *) dci_pdu)->TPC = TPC;
        ((DCI0_5MHz_TDD_1_6_t *) dci_pdu)->mcs = mcs;
        ((DCI0_5MHz_TDD_1_6_t *) dci_pdu)->ndi = ndi;
        ((DCI0_5MHz_TDD_1_6_t *) dci_pdu)->rballoc = rballoc;
        ((DCI0_5MHz_TDD_1_6_t *) dci_pdu)->hopping = hopping;
        ((DCI0_5MHz_TDD_1_6_t *) dci_pdu)->type = 0;
        ((DCI0_5MHz_TDD_1_6_t *) dci_pdu)->padding = 0;
        dci_alloc->dci_length = sizeof_DCI0_5MHz_TDD_1_6_t;
      } else {
        ((DCI0_5MHz_FDD_t *) dci_pdu)->cqi_req = cqi_req;
        ((DCI0_5MHz_FDD_t *) dci_pdu)->cshift = cshift;
        ((DCI0_5MHz_FDD_t *) dci_pdu)->TPC = TPC;
        ((DCI0_5MHz_FDD_t *) dci_pdu)->mcs = mcs;
        ((DCI0_5MHz_FDD_t *) dci_pdu)->ndi = ndi;
        ((DCI0_5MHz_FDD_t *) dci_pdu)->rballoc = rballoc;
        ((DCI0_5MHz_FDD_t *) dci_pdu)->hopping = hopping;
        ((DCI0_5MHz_FDD_t *) dci_pdu)->type = 0;
        ((DCI0_5MHz_FDD_t *) dci_pdu)->padding = 0;
        dci_alloc->dci_length = sizeof_DCI0_5MHz_FDD_t;
      }

      break;

    case 50:
      if (frame_parms->frame_type == TDD) {
        ((DCI0_10MHz_TDD_1_6_t *) dci_pdu)->cqi_req = cqi_req;
        ((DCI0_10MHz_TDD_1_6_t *) dci_pdu)->dai = dai;
        ((DCI0_10MHz_TDD_1_6_t *) dci_pdu)->cshift = cshift;
        ((DCI0_10MHz_TDD_1_6_t *) dci_pdu)->TPC = TPC;
        ((DCI0_10MHz_TDD_1_6_t *) dci_pdu)->mcs = mcs;
        ((DCI0_10MHz_TDD_1_6_t *) dci_pdu)->ndi = ndi;
        ((DCI0_10MHz_TDD_1_6_t *) dci_pdu)->rballoc = rballoc;
        ((DCI0_10MHz_TDD_1_6_t *) dci_pdu)->hopping = hopping;
        ((DCI0_10MHz_TDD_1_6_t *) dci_pdu)->type = 0;
        ((DCI0_10MHz_TDD_1_6_t *) dci_pdu)->padding = 0;
        dci_alloc->dci_length = sizeof_DCI0_10MHz_TDD_1_6_t;
      } else {
        ((DCI0_10MHz_FDD_t *) dci_pdu)->cqi_req = cqi_req;
        ((DCI0_10MHz_FDD_t *) dci_pdu)->cshift = cshift;
        ((DCI0_10MHz_FDD_t *) dci_pdu)->TPC = TPC;
        ((DCI0_10MHz_FDD_t *) dci_pdu)->mcs = mcs;
        ((DCI0_10MHz_FDD_t *) dci_pdu)->ndi = ndi;
        ((DCI0_10MHz_FDD_t *) dci_pdu)->rballoc = rballoc;
        ((DCI0_10MHz_FDD_t *) dci_pdu)->hopping = hopping;
        ((DCI0_10MHz_FDD_t *) dci_pdu)->type = 0;
        ((DCI0_10MHz_FDD_t *) dci_pdu)->padding = 0;
        dci_alloc->dci_length = sizeof_DCI0_10MHz_FDD_t;
      }

      break;

    case 100:
      if (frame_parms->frame_type == TDD) {
        ((DCI0_20MHz_TDD_1_6_t *) dci_pdu)->cqi_req = cqi_req;
        ((DCI0_20MHz_TDD_1_6_t *) dci_pdu)->dai = dai;
        ((DCI0_20MHz_TDD_1_6_t *) dci_pdu)->cshift = cshift;
        ((DCI0_20MHz_TDD_1_6_t *) dci_pdu)->TPC = TPC;
        ((DCI0_20MHz_TDD_1_6_t *) dci_pdu)->mcs = mcs;
        ((DCI0_20MHz_TDD_1_6_t *) dci_pdu)->ndi = ndi;
        ((DCI0_20MHz_TDD_1_6_t *) dci_pdu)->rballoc = rballoc;
        ((DCI0_20MHz_TDD_1_6_t *) dci_pdu)->hopping = hopping;
        ((DCI0_20MHz_TDD_1_6_t *) dci_pdu)->type = 0;
        ((DCI0_20MHz_TDD_1_6_t *) dci_pdu)->padding = 0;
        dci_alloc->dci_length = sizeof_DCI0_20MHz_TDD_1_6_t;
      } else {
        ((DCI0_20MHz_FDD_t *) dci_pdu)->cqi_req = cqi_req;
        ((DCI0_20MHz_FDD_t *) dci_pdu)->cshift = cshift;
        ((DCI0_20MHz_FDD_t *) dci_pdu)->TPC = TPC;
        ((DCI0_20MHz_FDD_t *) dci_pdu)->mcs = mcs;
        ((DCI0_20MHz_FDD_t *) dci_pdu)->ndi = ndi;
        ((DCI0_20MHz_FDD_t *) dci_pdu)->rballoc = rballoc;
        ((DCI0_20MHz_FDD_t *) dci_pdu)->hopping = hopping;
        ((DCI0_20MHz_FDD_t *) dci_pdu)->type = 0;
        ((DCI0_20MHz_FDD_t *) dci_pdu)->padding = 0;
        dci_alloc->dci_length = sizeof_DCI0_20MHz_FDD_t;
      }

      //printf("eNB: rb_alloc (20 MHz dci) %d\n",rballoc);
      break;

    default:
      LOG_E (PHY, "Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
      DevParam (frame_parms->N_RB_DL, 0, 0);
      break;
  }

  if(frame_parms->frame_type == TDD) {
    UE_id = find_ulsch(pdu->dci_pdu_rel8.rnti, eNB,SEARCH_EXIST_OR_FREE);

    if(UE_id != -1) {
      eNB->ulsch[UE_id]->harq_processes[pdu->dci_pdu_rel8.harq_pid]->V_UL_DAI = dai +1;
    }
  }
}

int get_narrowband_index(int N_RB_UL,int rb) {
  switch (N_RB_UL) {
    case 6: // 6 PRBs, N_NB=1, i_0=0
    case 25: // 25 PRBs, N_NB=4, i_0=0
      return(rb/6);
      break;

    case 50: // 50 PRBs, N_NB=8, i_0=1
    case 75: // 75 PRBs, N_NB=12, i_0=1
    case 15: // 15 PRBs, N_NB=2, i_0=1
      AssertFatal(rb>=1,"rb %d is not possible for %d PRBs\n",rb,N_RB_UL);
      return((rb-1)/6);
      break;

    case 100: // 100 PRBs, N_NB=16, i_0=2
      AssertFatal(rb>=2,"rb %d is not possible for %d PRBs\n",rb,N_RB_UL);
      return(rb-2/6);
      break;

    default:
      AssertFatal(1==0,"Impossible N_RB_UL %d\n",N_RB_UL);
      break;
  }
}

void fill_ulsch(PHY_VARS_eNB *eNB,int UE_id,nfapi_ul_config_ulsch_pdu *ulsch_pdu,int frame,int subframe) {
  uint8_t harq_pid;
  //uint8_t UE_id;
  boolean_t new_ulsch = (find_ulsch(ulsch_pdu->ulsch_pdu_rel8.rnti,eNB,SEARCH_EXIST)==-1) ? TRUE : FALSE;
  //AssertFatal((UE_id=find_ulsch(ulsch_pdu->ulsch_pdu_rel8.rnti,eNB,SEARCH_EXIST_OR_FREE))>=0,
  //        "No existing/free UE ULSCH for rnti %x\n",ulsch_pdu->ulsch_pdu_rel8.rnti);
  LTE_eNB_ULSCH_t *ulsch=eNB->ulsch[UE_id];
  LTE_DL_FRAME_PARMS *frame_parms = &eNB->frame_parms;
  int use_srs = 0;
  harq_pid = ulsch_pdu->ulsch_pdu_rel8.harq_process_number;
  ulsch->harq_mask |= 1 << harq_pid;
  LOG_D(PHY,"Filling ULSCH : ue_type %d, harq_pid %d\n",ulsch->ue_type,harq_pid);
  ulsch->ue_type = ulsch_pdu->ulsch_pdu_rel13.ue_type;
  AssertFatal(harq_pid ==0 || ulsch->ue_type == NOCE, "Harq PID is not zero for BL/CE UE\n");

  if(ulsch_pdu->ulsch_pdu_rel13.repetition_number >1)	// Fill the Harq process parameters in the first Rep only
  {
	  return;
  }

  ulsch->harq_processes[harq_pid]->frame                                 = frame;
  ulsch->harq_processes[harq_pid]->subframe                              = subframe;
  ulsch->harq_processes[harq_pid]->handled                               = 0;
  ulsch->harq_processes[harq_pid]->repetition_number                     = ulsch_pdu->ulsch_pdu_rel13.repetition_number ;
  ulsch->harq_processes[harq_pid]->total_number_of_repetitions           = ulsch_pdu->ulsch_pdu_rel13.total_number_of_repetitions ;
  ulsch->harq_processes[harq_pid]->first_rb                              = ulsch_pdu->ulsch_pdu_rel8.resource_block_start;
  ulsch->harq_processes[harq_pid]->nb_rb                                 = ulsch_pdu->ulsch_pdu_rel8.number_of_resource_blocks;
  ulsch->harq_processes[harq_pid]->dci_alloc                             = 1;
  ulsch->harq_processes[harq_pid]->rar_alloc                             = 0;
  ulsch->harq_processes[harq_pid]->n_DMRS                                = ulsch_pdu->ulsch_pdu_rel8.cyclic_shift_2_for_drms;
  ulsch->harq_processes[harq_pid]->Nsymb_pusch                           = 12-(frame_parms->Ncp<<1)-(use_srs==0?0:1);
  ulsch->harq_processes[harq_pid]->srs_active                            = use_srs;

  //Mapping of cyclic shift field in DCI format0 to n_DMRS2 (3GPP 36.211, Table 5.5.2.1.1-1)
  if(ulsch->harq_processes[harq_pid]->n_DMRS == 0)
    ulsch->harq_processes[harq_pid]->n_DMRS2 = 0;
  else if(ulsch->harq_processes[harq_pid]->n_DMRS == 1)
    ulsch->harq_processes[harq_pid]->n_DMRS2 = 6;
  else if(ulsch->harq_processes[harq_pid]->n_DMRS == 2)
    ulsch->harq_processes[harq_pid]->n_DMRS2 = 3;
  else if(ulsch->harq_processes[harq_pid]->n_DMRS == 3)
    ulsch->harq_processes[harq_pid]->n_DMRS2 = 4;
  else if(ulsch->harq_processes[harq_pid]->n_DMRS == 4)
    ulsch->harq_processes[harq_pid]->n_DMRS2 = 2;
  else if(ulsch->harq_processes[harq_pid]->n_DMRS == 5)
    ulsch->harq_processes[harq_pid]->n_DMRS2 = 8;
  else if(ulsch->harq_processes[harq_pid]->n_DMRS == 6)
    ulsch->harq_processes[harq_pid]->n_DMRS2 = 10;
  else if(ulsch->harq_processes[harq_pid]->n_DMRS == 7)
    ulsch->harq_processes[harq_pid]->n_DMRS2 = 9;

  LOG_D(PHY,
        "[eNB %d][PUSCH %d] Frame %d, Subframe %d Programming PUSCH with n_DMRS2 %d (cshift %d) ulsch:ndi:%d ulsch_pdu:ndi:%d new_ulsch:%d status:%d ulsch_pdu:rvidx:%d ulsch_pdu->ulsch_pdu_rel8.size %d\n",
        eNB->Mod_id,harq_pid,frame,subframe,
        ulsch->harq_processes[harq_pid]->n_DMRS2,
        ulsch->harq_processes[harq_pid]->n_DMRS,
        ulsch->harq_processes[harq_pid]->ndi, ulsch_pdu->ulsch_pdu_rel8.new_data_indication, new_ulsch, ulsch->harq_processes[harq_pid]->status,
        ulsch_pdu->ulsch_pdu_rel8.redundancy_version,
        ulsch_pdu->ulsch_pdu_rel8.size);
  ulsch->harq_processes[harq_pid]->rvidx = ulsch_pdu->ulsch_pdu_rel8.redundancy_version;

  if(ulsch_pdu->ulsch_pdu_rel8.modulation_type!=0)
    ulsch->harq_processes[harq_pid]->Qm    = ulsch_pdu->ulsch_pdu_rel8.modulation_type;

  // Set O_ACK to 0 by default, will be set of DLSCH is scheduled and needs to be
  ulsch->harq_processes[harq_pid]->O_ACK         = 0;

  if ((ulsch->harq_processes[harq_pid]->status == SCH_IDLE) ||
      (ulsch->harq_processes[harq_pid]->ndi    != ulsch_pdu->ulsch_pdu_rel8.new_data_indication) ||
      (new_ulsch == TRUE)) {
    ulsch->harq_processes[harq_pid]->status        = ACTIVE;
    ulsch->harq_processes[harq_pid]->TBS           = ulsch_pdu->ulsch_pdu_rel8.size<<3;
    ulsch->harq_processes[harq_pid]->Msc_initial   = 12*ulsch_pdu->ulsch_pdu_rel8.number_of_resource_blocks;
    ulsch->harq_processes[harq_pid]->Nsymb_initial = ulsch->harq_processes[harq_pid]->Nsymb_pusch;
    ulsch->harq_processes[harq_pid]->round         = 0;
    ulsch->harq_processes[harq_pid]->ndi           = ulsch_pdu->ulsch_pdu_rel8.new_data_indication;
    // note here, the CQI bits need to be kept constant as in initial transmission
    // set to 0 in initial transmission, and don't touch them during retransmissions
    // will be set if MAC has activated ULSCH_CQI_RI_PDU or ULSCH_CQI_HARQ_RI_PDU
    ulsch->harq_processes[harq_pid]->Or1           = 0;
    ulsch->harq_processes[harq_pid]->Or2           = 0;
  } else {
    ulsch->harq_processes[harq_pid]->round++;
    ulsch->harq_processes[harq_pid]->TBS           = ulsch_pdu->ulsch_pdu_rel8.size<<3;
    ulsch->harq_processes[harq_pid]->Msc_initial   = 12*ulsch_pdu->ulsch_pdu_rel8.number_of_resource_blocks;
    ulsch->harq_processes[harq_pid]->Or1           = 0;
    ulsch->harq_processes[harq_pid]->Or2           = 0;
  }

  ulsch->rnti = ulsch_pdu->ulsch_pdu_rel8.rnti;
  LOG_D(PHY,"Filling ULSCH %x (UE_id %d) (new_ulsch %d) for Frame %d, Subframe %d : harq_pid %d, status %d, handled %d, first_rb %d, nb_rb %d, rvidx %d, Qm %d, TBS %d, round %d \n",
        ulsch->rnti,
        UE_id,
        new_ulsch,
        frame,
        subframe,
        harq_pid,
        ulsch->harq_processes[harq_pid]->status,
        ulsch->harq_processes[harq_pid]->handled,
        ulsch->harq_processes[harq_pid]->first_rb,
        ulsch->harq_processes[harq_pid]->nb_rb,
        ulsch->harq_processes[harq_pid]->rvidx,
        ulsch->harq_processes[harq_pid]->Qm,
        ulsch->harq_processes[harq_pid]->TBS,
        ulsch->harq_processes[harq_pid]->round);
}



int get_first_rb_in_narrowband(int N_RB_UL,
                               int rb) {
  switch (N_RB_UL) {
    case 6: // 6 PRBs, N_NB=1, i_0=0
    case 25: // 25 PRBs, N_NB=4, i_0=0
      return(rb - 6*(rb/6));
      break;

    case 50: // 50 PRBs, N_NB=8, i_0=1
    case 75: // 75 PRBs, N_NB=12, i_0=1
    case 15: // 15 PRBs, N_NB=2, i_0=1
      AssertFatal(rb>=1,"rb %d is not possible for %d PRBs\n",rb,N_RB_UL);
      return(rb-1-(6*((rb-1)/6)));
      break;

    case 100: // 100 PRBs, N_NB=16, i_0=2
      AssertFatal(rb>=2,"rb %d is not possible for %d PRBs\n",rb,N_RB_UL);
      return(rb-2-(6*((rb-2)/6)));
      break;

    default:
      AssertFatal(1==0,"Impossible N_RB_UL %d\n",N_RB_UL);
      break;
  }
}

void fill_mpdcch_dci0 (PHY_VARS_eNB *eNB,
                       L1_rxtx_proc_t *proc,
                       mDCI_ALLOC_t *dci_alloc,
                       nfapi_hi_dci0_mpdcch_dci_pdu *pdu) {
  LTE_DL_FRAME_PARMS *frame_parms = &eNB->frame_parms;
  nfapi_hi_dci0_mpdcch_dci_pdu_rel13_t *rel13 = &pdu->mpdcch_dci_pdu_rel13;
  uint32_t        cqi_req = rel13->csi_request;
  uint32_t        TPC = rel13->tpc;
  uint32_t        mcs = rel13->mcs;
  uint32_t        hopping = rel13->frequency_hopping_flag;
  uint32_t        narrowband = get_narrowband_index(frame_parms->N_RB_UL,rel13->resource_block_start);
  uint32_t        rballoc = computeRIV (6,
                                        get_first_rb_in_narrowband(frame_parms->N_RB_UL,rel13->resource_block_start),
                                        rel13->number_of_resource_blocks);
  uint32_t        ndi = rel13->new_data_indication;
#ifdef T_TRACER
  T (T_ENB_PHY_ULSCH_UE_DCI, T_INT (eNB->Mod_id), T_INT (proc->frame_tx), T_INT (proc->subframe_tx),
     T_INT (rel13->rnti), T_INT (((proc->frame_tx * 10 + proc->subframe_tx + 4) % 8) /* TODO: correct harq pid */ ),
     T_INT (mcs), T_INT (-1 /* TODO: remove round? */ ),
     T_INT (rel13->resource_block_start),
     T_INT (rel13->number_of_resource_blocks),
     T_INT (get_TBS_UL (mcs, rel13->number_of_resource_blocks) * 8), T_INT (rel13->aggreagation_level), T_INT (rel13->ecce_index));
#endif
  void           *dci_pdu = (void *) dci_alloc->dci_pdu;
  AssertFatal(rel13->ce_mode == 1 && rel13->dci_format == 4, "dci format 5 (CE_modeB) not supported yet\n");
  LOG_D (PHY, "Filling DCI6-0A with cqi %d, mcs %d, hopping %d, rballoc %x (%d,%d) ndi %d TPC %d\n", cqi_req,
         mcs, hopping, rballoc, rel13->resource_block_start, rel13->number_of_resource_blocks, ndi, TPC);
  dci_alloc->format = format6_0A;
  dci_alloc->firstCCE = rel13->ecce_index;
  dci_alloc->L = rel13->aggreagation_level;
  dci_alloc->rnti = rel13->rnti;
  dci_alloc->harq_pid = rel13->harq_process;
  dci_alloc->narrowband = rel13->mpdcch_narrowband;
  dci_alloc->number_of_prb_pairs = rel13->number_of_prb_pairs;
  dci_alloc->resource_block_assignment = rel13->resource_block_assignment;
  dci_alloc->transmission_type = rel13->mpdcch_transmission_type;
  dci_alloc->start_symbol = rel13->start_symbol;
  dci_alloc->ce_mode = rel13->ce_mode;
  dci_alloc->dmrs_scrambling_init = rel13->drms_scrambling_init;
  dci_alloc->i0 = rel13->initial_transmission_sf_io;

  switch (frame_parms->N_RB_DL) {
    case 6:
      if (frame_parms->frame_type == TDD) {
        AssertFatal(1==0,"TDD not supported for eMTC yet\n");
      } else {
        AssertFatal(1==0,"6 PRBS not supported for eMTC\n");
      }

      break;

    case 25:
      if (frame_parms->frame_type == TDD) {
        AssertFatal(1==0,"TDD not supported for eMTC yet\n");
      } else {
        dci_alloc->dci_length = sizeof_DCI6_0A_5MHz_t;
        ((DCI6_0A_5MHz_t *) dci_pdu)->type = 0;
        ((DCI6_0A_5MHz_t *) dci_pdu)->hopping = hopping;
        ((DCI6_0A_5MHz_t *) dci_pdu)->rballoc = rballoc;
        ((DCI6_0A_5MHz_t *) dci_pdu)->narrowband = narrowband;
        ((DCI6_0A_5MHz_t *) dci_pdu)->mcs = mcs;
        ((DCI6_0A_5MHz_t *) dci_pdu)->rep = rel13->pusch_repetition_levels;
        ((DCI6_0A_5MHz_t *) dci_pdu)->harq_pid = rel13->harq_process;
        ((DCI6_0A_5MHz_t *) dci_pdu)->ndi = ndi;
        ((DCI6_0A_5MHz_t *) dci_pdu)->rv_idx = rel13->redudency_version;
        ((DCI6_0A_5MHz_t *) dci_pdu)->TPC = TPC;
        ((DCI6_0A_5MHz_t *) dci_pdu)->csi_req = cqi_req;
        ((DCI6_0A_5MHz_t *) dci_pdu)->srs_req = rel13->srs_request;
        ((DCI6_0A_5MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
      }

      break;

    case 50:
      if (frame_parms->frame_type == TDD) {
        AssertFatal(1==0,"TDD not supported for eMTC yet\n");
      } else {
        dci_alloc->dci_length = sizeof_DCI6_0A_10MHz_t;
        ((DCI6_0A_10MHz_t *) dci_pdu)->type = 0;
        ((DCI6_0A_10MHz_t *) dci_pdu)->hopping = hopping;
        ((DCI6_0A_10MHz_t *) dci_pdu)->rballoc = rballoc;
        ((DCI6_0A_10MHz_t *) dci_pdu)->narrowband = narrowband;
        ((DCI6_0A_10MHz_t *) dci_pdu)->mcs = mcs;
        ((DCI6_0A_10MHz_t *) dci_pdu)->rep = rel13->pusch_repetition_levels;
        ((DCI6_0A_10MHz_t *) dci_pdu)->harq_pid = rel13->harq_process;
        ((DCI6_0A_10MHz_t *) dci_pdu)->ndi = ndi;
        ((DCI6_0A_10MHz_t *) dci_pdu)->rv_idx = rel13->redudency_version;
        ((DCI6_0A_10MHz_t *) dci_pdu)->TPC = TPC;
        ((DCI6_0A_10MHz_t *) dci_pdu)->csi_req = cqi_req;
        ((DCI6_0A_10MHz_t *) dci_pdu)->srs_req = rel13->srs_request;
        ((DCI6_0A_10MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
        LOG_D(PHY,
              "Frame %d, Subframe %d : Programming Format 6-0A DCI, type %d, hopping %d, narrowband %d, rballoc %x, mcs %d, rep %d, harq_pid %d, ndi %d, rv %d, TPC %d, csi_req %d, srs_req %d, dci_rep r%d => %x\n",
              proc->frame_tx,proc->subframe_tx,
              ((DCI6_0A_10MHz_t *) dci_pdu)->type,
              ((DCI6_0A_10MHz_t *) dci_pdu)->hopping,
              ((DCI6_0A_10MHz_t *) dci_pdu)->narrowband,
              ((DCI6_0A_10MHz_t *) dci_pdu)->rballoc,
              ((DCI6_0A_10MHz_t *) dci_pdu)->mcs,
              ((DCI6_0A_10MHz_t *) dci_pdu)->rep,
              ((DCI6_0A_10MHz_t *) dci_pdu)->harq_pid,
              ((DCI6_0A_10MHz_t *) dci_pdu)->ndi,
              ((DCI6_0A_10MHz_t *) dci_pdu)->rv_idx,
              ((DCI6_0A_10MHz_t *) dci_pdu)->TPC,
              ((DCI6_0A_10MHz_t *) dci_pdu)->csi_req,
              ((DCI6_0A_10MHz_t *) dci_pdu)->srs_req,
              ((DCI6_0A_10MHz_t *) dci_pdu)->dci_rep,
              ((uint32_t *)dci_pdu)[0]);
      }

      break;

    case 100:
      if (frame_parms->frame_type == TDD) {
        AssertFatal(1==0,"TDD not supported for eMTC yet\n");
      } else {
        dci_alloc->dci_length = sizeof_DCI6_0A_20MHz_t;
        ((DCI6_0A_20MHz_t *) dci_pdu)->type = 0;
        ((DCI6_0A_20MHz_t *) dci_pdu)->hopping = hopping;
        ((DCI6_0A_20MHz_t *) dci_pdu)->rballoc = rballoc;
        ((DCI6_0A_20MHz_t *) dci_pdu)->narrowband = narrowband;
        ((DCI6_0A_20MHz_t *) dci_pdu)->mcs = rel13->mcs;
        ((DCI6_0A_20MHz_t *) dci_pdu)->rep = rel13->pusch_repetition_levels;
        ((DCI6_0A_20MHz_t *) dci_pdu)->harq_pid = rel13->harq_process;
        ((DCI6_0A_20MHz_t *) dci_pdu)->ndi = ndi;
        ((DCI6_0A_20MHz_t *) dci_pdu)->rv_idx = rel13->redudency_version;
        ((DCI6_0A_20MHz_t *) dci_pdu)->TPC = TPC;
        ((DCI6_0A_20MHz_t *) dci_pdu)->csi_req = cqi_req;
        ((DCI6_0A_20MHz_t *) dci_pdu)->srs_req = rel13->srs_request;
        ((DCI6_0A_20MHz_t *) dci_pdu)->dci_rep = rel13->dci_subframe_repetition_number;
      }

      //printf("eNB: rb_alloc (20 MHz dci) %d\n",rballoc);
      break;

    default:
      LOG_E (PHY, "Invalid N_RB_DL %d\n", frame_parms->N_RB_DL);
      DevParam (frame_parms->N_RB_DL, 0, 0);
      break;
  }
}

