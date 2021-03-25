/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include "executables/nr-uesoftmodem.h"
#include "PHY/phy_extern_nr_ue.h"
#include "PHY/INIT/phy_init.h"
#include "NR_MAC_UE/mac_proto.h"
#include "RRC/NR_UE/rrc_proto.h"
#include "SCHED_NR_UE/phy_frame_config_nr.h"
#include "SCHED_NR_UE/defs.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"

/*
 *  NR SLOT PROCESSING SEQUENCE
 *
 *  Processing occurs with following steps for connected mode:
 *
 *  - Rx samples for a slot are received,
 *  - PDCCH processing (including DCI extraction for downlink and uplink),
 *  - PDSCH processing (including transport blocks decoding),
 *  - PUCCH/PUSCH (transmission of acknowledgements, CSI, ... or data).
 *
 *  Time between reception of the slot and related transmission depends on UE processing performance.
 *  It is defined by the value NR_UE_CAPABILITY_SLOT_RX_TO_TX.
 *
 *  In NR, network gives the duration between Rx slot and Tx slot in the DCI:
 *  - for reception of a PDSCH and its associated acknowledgment slot (with a PUCCH or a PUSCH),
 *  - for reception of an uplink grant and its associated PUSCH slot.
 *
 *  So duration between reception and it associated transmission depends on its transmission slot given in the DCI.
 *  NR_UE_CAPABILITY_SLOT_RX_TO_TX means the minimum duration but higher duration can be given by the network because UE can support it.
 *
 *                                                                                                    Slot k
 *                                                                                  -------+------------+--------
 *                Frame                                                                    | Tx samples |
 *                Subframe                                                                 |   buffer   |
 *                Slot n                                                            -------+------------+--------
 *       ------ +------------+--------                                                     |
 *              | Rx samples |                                                             |
 *              |   buffer   |                                                             |
 *       -------+------------+--------                                                     |
 *                           |                                                             |
 *                           V                                                             |
 *                           +------------+                                                |
 *                           |   PDCCH    |                                                |
 *                           | processing |                                                |
 *                           +------------+                                                |
 *                           |            |                                                |
 *                           |            v                                                |
 *                           |            +------------+                                   |
 *                           |            |   PDSCH    |                                   |
 *                           |            | processing | decoding result                   |
 *                           |            +------------+    -> ACK/NACK of PDSCH           |
 *                           |                         |                                   |
 *                           |                         v                                   |
 *                           |                         +-------------+------------+        |
 *                           |                         | PUCCH/PUSCH | Tx samples |        |
 *                           |                         |  processing | transfer   |        |
 *                           |                         +-------------+------------+        |
 *                           |                                                             |
 *                           |/___________________________________________________________\|
 *                            \  duration between reception and associated transmission   /
 *
 * Remark: processing is done slot by slot, it can be distribute on different threads which are executed in parallel.
 * This is an architecture optimization in order to cope with real time constraints.
 * By example, for LTE, subframe processing is spread over 4 different threads.
 *
 */

typedef enum {
  pss = 0,
  pbch = 1,
  si = 2
} sync_mode_t;

void init_nr_ue_vars(PHY_VARS_NR_UE *ue,
                     uint8_t UE_id,
                     uint8_t abstraction_flag)
{

  int nb_connected_gNB = 1, gNB_id;

  ue->Mod_id      = UE_id;
  ue->mac_enabled = 1;
  ue->if_inst     = nr_ue_if_module_init(0);

  // Setting UE mode to NOT_SYNCHED by default
  for (gNB_id = 0; gNB_id < nb_connected_gNB; gNB_id++){
    ue->UE_mode[gNB_id] = NOT_SYNCHED;
    ue->prach_resources[gNB_id] = (NR_PRACH_RESOURCES_t *)malloc16_clear(sizeof(NR_PRACH_RESOURCES_t));
  }

  // initialize all signal buffers
  init_nr_ue_signal(ue, nb_connected_gNB, abstraction_flag);

  // intialize transport
  init_nr_ue_transport(ue, abstraction_flag);

  // init N_TA offset
  init_N_TA_offset(ue);
}

/*!
 * It performs band scanning and synchonization.
 * \param arg is a pointer to a \ref PHY_VARS_NR_UE structure.
 */

typedef struct syncData_s {
  UE_nr_rxtx_proc_t proc;
  PHY_VARS_NR_UE *UE;
} syncData_t;

static void UE_synch(void *arg) {
  syncData_t *syncD=(syncData_t *) arg;
  int i, hw_slot_offset;
  PHY_VARS_NR_UE *UE = syncD->UE;
  sync_mode_t sync_mode = pbch;
  //int CC_id = UE->CC_id;
  static int freq_offset=0;
  UE->is_synchronized = 0;

  if (UE->UE_scan == 0) {

    #ifdef FR2_TEST
    // Overwrite DL frequency (for FR2 testing)
    if (downlink_frequency[0][0]!=0){
       UE->frame_parms.dl_CarrierFreq = downlink_frequency[0][0];
       UE->frame_parms.ul_CarrierFreq = downlink_frequency[0][0];
    }
    #endif

    for (i=0; i<openair0_cfg[UE->rf_map.card].rx_num_channels; i++) {

      LOG_I( PHY, "[SCHED][UE] Check absolute frequency DL %f, UL %f (RF card %d, oai_exit %d, channel %d, rx_num_channels %d)\n",
        openair0_cfg[UE->rf_map.card].rx_freq[UE->rf_map.chain+i],
        openair0_cfg[UE->rf_map.card].tx_freq[UE->rf_map.chain+i],
        UE->rf_map.card,
        oai_exit,
        i,
        openair0_cfg[0].rx_num_channels);

    }

    sync_mode = pbch;
  } else {
    LOG_E(PHY,"Fixme!\n");
    /*
    for (i=0; i<openair0_cfg[UE->rf_map.card].rx_num_channels; i++) {
      downlink_frequency[UE->rf_map.card][UE->rf_map.chain+i] = bands_to_scan.band_info[CC_id].dl_min;
      uplink_frequency_offset[UE->rf_map.card][UE->rf_map.chain+i] =
        bands_to_scan.band_info[CC_id].ul_min-bands_to_scan.band_info[CC_id].dl_min;
      openair0_cfg[UE->rf_map.card].rx_freq[UE->rf_map.chain+i] = downlink_frequency[CC_id][i];
      openair0_cfg[UE->rf_map.card].tx_freq[UE->rf_map.chain+i] =
        downlink_frequency[CC_id][i]+uplink_frequency_offset[CC_id][i];
      openair0_cfg[UE->rf_map.card].rx_gain[UE->rf_map.chain+i] = UE->rx_total_gain_dB;
    }
    */
  }

  LOG_W(PHY, "Starting sync detection\n");

  switch (sync_mode) {
    /*
    case pss:
      LOG_I(PHY,"[SCHED][UE] Scanning band %d (%d), freq %u\n",bands_to_scan.band_info[current_band].band, current_band,bands_to_scan.band_info[current_band].dl_min+current_offset);
      //lte_sync_timefreq(UE,current_band,bands_to_scan.band_info[current_band].dl_min+current_offset);
      current_offset += 20000000; // increase by 20 MHz

      if (current_offset > bands_to_scan.band_info[current_band].dl_max-bands_to_scan.band_info[current_band].dl_min) {
        current_band++;
        current_offset=0;
      }

      if (current_band==bands_to_scan.nbands) {
        current_band=0;
        oai_exit=1;
      }

      for (i=0; i<openair0_cfg[UE->rf_map.card].rx_num_channels; i++) {
        downlink_frequency[UE->rf_map.card][UE->rf_map.chain+i] = bands_to_scan.band_info[current_band].dl_min+current_offset;
        uplink_frequency_offset[UE->rf_map.card][UE->rf_map.chain+i] = bands_to_scan.band_info[current_band].ul_min-bands_to_scan.band_info[0].dl_min + current_offset;
        openair0_cfg[UE->rf_map.card].rx_freq[UE->rf_map.chain+i] = downlink_frequency[CC_id][i];
        openair0_cfg[UE->rf_map.card].tx_freq[UE->rf_map.chain+i] = downlink_frequency[CC_id][i]+uplink_frequency_offset[CC_id][i];
        openair0_cfg[UE->rf_map.card].rx_gain[UE->rf_map.chain+i] = UE->rx_total_gain_dB;

        if (UE->UE_scan_carrier) {
          openair0_cfg[UE->rf_map.card].autocal[UE->rf_map.chain+i] = 1;
        }
      }

      break;
    */
    case pbch:
      LOG_I(PHY, "[UE thread Synch] Running Initial Synch (mode %d)\n",UE->mode);

      uint64_t dl_carrier, ul_carrier;
      double rx_gain_off = 0;
      nr_get_carrier_frequencies(&UE->frame_parms, &dl_carrier, &ul_carrier);

      if (nr_initial_sync(&syncD->proc, UE, 2) == 0) {
        freq_offset = UE->common_vars.freq_offset; // frequency offset computed with pss in initial sync
        hw_slot_offset = ((UE->rx_offset<<1) / UE->frame_parms.samples_per_subframe * UE->frame_parms.slots_per_subframe) +
                         round((float)((UE->rx_offset<<1) % UE->frame_parms.samples_per_subframe)/UE->frame_parms.samples_per_slot0);

        // rerun with new cell parameters and frequency-offset
        // todo: the freq_offset computed on DL shall be scaled before being applied to UL
        nr_rf_card_config(&openair0_cfg[UE->rf_map.card], rx_gain_off, ul_carrier, dl_carrier, freq_offset);

        LOG_I(PHY,"Got synch: hw_slot_offset %d, carrier off %d Hz, rxgain %f (DL %f Hz, UL %f Hz)\n",
              hw_slot_offset,
              freq_offset,
              openair0_cfg[UE->rf_map.card].rx_gain[0],
              openair0_cfg[UE->rf_map.card].rx_freq[0],
              openair0_cfg[UE->rf_map.card].tx_freq[0]);

        if (UE->mode != loop_through_memory) {
          UE->rfdevice.trx_set_freq_func(&UE->rfdevice,&openair0_cfg[0],0);
          //UE->rfdevice.trx_set_gains_func(&openair0,&openair0_cfg[0]);
          //UE->rfdevice.trx_stop_func(&UE->rfdevice);
          // sleep(1);
          /*if (UE->rfdevice.trx_start_func(&UE->rfdevice) != 0 ) {
            LOG_E(HW,"Could not start the device\n");
            oai_exit=1;
            }*/
        }

        if (UE->UE_scan_carrier == 1) {
          UE->UE_scan_carrier = 0;
        } else {
          UE->is_synchronized = 1;
        }
      } else {

        if (UE->UE_scan_carrier == 1) {

          if (freq_offset >= 0)
            freq_offset += 100;

          freq_offset *= -1;

          nr_rf_card_config(&openair0_cfg[UE->rf_map.card], rx_gain_off, ul_carrier, dl_carrier, freq_offset);

          LOG_I(PHY, "Initial sync failed: trying carrier off %d Hz\n", freq_offset);

          if (UE->mode != loop_through_memory)
            UE->rfdevice.trx_set_freq_func(&UE->rfdevice,&openair0_cfg[0],0);
        }

        break;

      case si:
      default:
        break;
      }
  }
}

void processSlotTX( PHY_VARS_NR_UE *UE, UE_nr_rxtx_proc_t *proc) {
  fapi_nr_config_request_t *cfg = &UE->nrUE_config;
  int tx_slot_type = nr_ue_slot_select(cfg, proc->frame_tx, proc->nr_slot_tx);
  uint8_t gNB_id = 0;

  if (tx_slot_type == NR_UPLINK_SLOT || tx_slot_type == NR_MIXED_SLOT){

    // trigger L2 to run ue_scheduler thru IF module
    // [TODO] mapping right after NR initial sync
    if(UE->if_inst != NULL && UE->if_inst->ul_indication != NULL) {
      nr_uplink_indication_t ul_indication;
      memset((void*)&ul_indication, 0, sizeof(ul_indication));

      ul_indication.module_id = UE->Mod_id;
      ul_indication.gNB_index = gNB_id;
      ul_indication.cc_id     = UE->CC_id;
      ul_indication.frame_rx  = proc->frame_rx;
      ul_indication.slot_rx   = proc->nr_slot_rx;
      ul_indication.frame_tx  = proc->frame_tx;
      ul_indication.slot_tx   = proc->nr_slot_tx;
      ul_indication.thread_id = proc->thread_id;

      UE->if_inst->ul_indication(&ul_indication);
    }

    if (UE->mode != loop_through_memory) {
      phy_procedures_nrUE_TX(UE,proc,0);
    }
  }
}

void processSlotRX( PHY_VARS_NR_UE *UE, UE_nr_rxtx_proc_t *proc) {

  fapi_nr_config_request_t *cfg = &UE->nrUE_config;
  int rx_slot_type = nr_ue_slot_select(cfg, proc->frame_rx, proc->nr_slot_rx);
  uint8_t gNB_id = 0;

  if (rx_slot_type == NR_DOWNLINK_SLOT || rx_slot_type == NR_MIXED_SLOT){

    if(UE->if_inst != NULL && UE->if_inst->dl_indication != NULL) {
      nr_downlink_indication_t dl_indication;
      nr_fill_dl_indication(&dl_indication, NULL, NULL, proc, UE, gNB_id);
      UE->if_inst->dl_indication(&dl_indication, NULL);
    }

  // Process Rx data for one sub-frame
#ifdef UE_SLOT_PARALLELISATION
    phy_procedures_slot_parallelization_nrUE_RX( UE, proc, 0, 0, 1, no_relay, NULL );
#else
    uint64_t a=rdtsc();
    phy_procedures_nrUE_RX(UE, proc, gNB_id, get_nrUE_params()->nr_dlsch_parallel);
    LOG_D(PHY, "In %s: slot %d, time %lu\n", __FUNCTION__, proc->nr_slot_rx, (rdtsc()-a)/3500);
#endif

    if(IS_SOFTMODEM_NOS1){
      NR_UE_MAC_INST_t *mac = get_mac_inst(0);
      protocol_ctxt_t ctxt;
      PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, UE->Mod_id, ENB_FLAG_NO, mac->crnti, proc->frame_rx, proc->nr_slot_rx, 0);
      pdcp_run(&ctxt);
    }
  }

}

/*!
 * \brief This is the UE thread for RX subframe n and TX subframe n+4.
 * This thread performs the phy_procedures_UE_RX() on every received slot.
 * then, if TX is enabled it performs TX for n+4.
 * \param arg is a pointer to a \ref PHY_VARS_NR_UE structure.
 * \returns a pointer to an int. The storage is not on the heap and must not be freed.
 */

typedef struct processingData_s {
  UE_nr_rxtx_proc_t proc;
  PHY_VARS_NR_UE    *UE;
}  processingData_t;

void UE_processing(void *arg) {
  processingData_t *rxtxD = (processingData_t *) arg;
  UE_nr_rxtx_proc_t *proc = &rxtxD->proc;
  PHY_VARS_NR_UE    *UE   = rxtxD->UE;
  int slot_tx = proc->nr_slot_tx;
  int frame_tx = proc->frame_tx;

  processSlotRX(UE, proc);
  processSlotTX(UE, proc);
  ue_ta_procedures(UE, slot_tx, frame_tx);

}

void dummyWrite(PHY_VARS_NR_UE *UE,openair0_timestamp timestamp, int writeBlockSize) {
  void *dummy_tx[UE->frame_parms.nb_antennas_tx];

  for (int i=0; i<UE->frame_parms.nb_antennas_tx; i++)
    dummy_tx[i]=malloc16_clear(writeBlockSize*4);

  AssertFatal( writeBlockSize ==
               UE->rfdevice.trx_write_func(&UE->rfdevice,
               timestamp,
               dummy_tx,
               writeBlockSize,
               UE->frame_parms.nb_antennas_tx,
               4),"");

  for (int i=0; i<UE->frame_parms.nb_antennas_tx; i++)
    free(dummy_tx[i]);
}

void readFrame(PHY_VARS_NR_UE *UE,  openair0_timestamp *timestamp, bool toTrash) {

  void *rxp[NB_ANTENNAS_RX];

  for(int x=0; x<20; x++) {  // two frames for initial sync
    for (int slot=0; slot<UE->frame_parms.slots_per_subframe; slot ++ ) {
      for (int i=0; i<UE->frame_parms.nb_antennas_rx; i++) {
        if (toTrash)
          rxp[i]=malloc16(UE->frame_parms.get_samples_per_slot(slot,&UE->frame_parms)*4);
        else
          rxp[i] = ((void *)&UE->common_vars.rxdata[i][0]) +
                   4*((x*UE->frame_parms.samples_per_subframe)+
                   UE->frame_parms.get_samples_slot_timestamp(slot,&UE->frame_parms,0));
      }
        
      AssertFatal( UE->frame_parms.get_samples_per_slot(slot,&UE->frame_parms) ==
                   UE->rfdevice.trx_read_func(&UE->rfdevice,
                   timestamp,
                   rxp,
                   UE->frame_parms.get_samples_per_slot(slot,&UE->frame_parms),
                   UE->frame_parms.nb_antennas_rx), "");

      if (IS_SOFTMODEM_RFSIM)
        dummyWrite(UE,*timestamp, UE->frame_parms.get_samples_per_slot(slot,&UE->frame_parms));
      if (toTrash)
        for (int i=0; i<UE->frame_parms.nb_antennas_rx; i++)
          free(rxp[i]);
    }
  }

}

void syncInFrame(PHY_VARS_NR_UE *UE, openair0_timestamp *timestamp) {

    LOG_I(PHY,"Resynchronizing RX by %d samples (mode = %d)\n",UE->rx_offset,UE->mode);

    *timestamp += UE->frame_parms.get_samples_per_slot(1,&UE->frame_parms);
    for ( int size=UE->rx_offset ; size > 0 ; size -= UE->frame_parms.samples_per_subframe ) {
      int unitTransfer=size>UE->frame_parms.samples_per_subframe ? UE->frame_parms.samples_per_subframe : size ;
      // we write before read becasue gNB waits for UE to write and both executions halt
      // this happens here as the read size is samples_per_subframe which is very much larger than samp_per_slot
      if (IS_SOFTMODEM_RFSIM) dummyWrite(UE,*timestamp, unitTransfer);
      AssertFatal(unitTransfer ==
                  UE->rfdevice.trx_read_func(&UE->rfdevice,
                                             timestamp,
                                             (void **)UE->common_vars.rxdata,
                                             unitTransfer,
                                             UE->frame_parms.nb_antennas_rx),"");
      *timestamp += unitTransfer; // this does not affect the read but needed for RFSIM write
    }

}

int computeSamplesShift(PHY_VARS_NR_UE *UE) {

  // compute TO compensation that should be applied for this frame
  if ( UE->rx_offset < UE->frame_parms.samples_per_frame/2  &&
       UE->rx_offset > 0 ) {
    //LOG_I(PHY,"!!!adjusting -1 samples!!!\n");
    return -1 ;
  }

  if ( UE->rx_offset > UE->frame_parms.samples_per_frame/2 &&
       UE->rx_offset < UE->frame_parms.samples_per_frame ) {
    //LOG_I(PHY,"!!!adjusting +1 samples!!!\n");
    return 1;
  }

  return 0;
}

static inline int get_firstSymSamp(uint16_t slot, NR_DL_FRAME_PARMS *fp) {
  if (fp->numerology_index == 0)
    return fp->nb_prefix_samples0 + fp->ofdm_symbol_size;
  int num_samples = (slot%(fp->slots_per_subframe/2)) ? fp->nb_prefix_samples : fp->nb_prefix_samples0;
  num_samples += fp->ofdm_symbol_size;
  return num_samples;
}

static inline int get_readBlockSize(uint16_t slot, NR_DL_FRAME_PARMS *fp) {
  int rem_samples = fp->get_samples_per_slot(slot, fp) - get_firstSymSamp(slot, fp);
  int next_slot_first_symbol = 0;
  if (slot < (fp->slots_per_frame-1))
    next_slot_first_symbol = get_firstSymSamp(slot+1, fp);
  return rem_samples + next_slot_first_symbol;
}

void *UE_thread(void *arg) {
  //this thread should be over the processing thread to keep in real time
  PHY_VARS_NR_UE *UE = (PHY_VARS_NR_UE *) arg;
  //  int tx_enabled = 0;
  openair0_timestamp timestamp, writeTimestamp;
  void *rxp[NB_ANTENNAS_RX], *txp[NB_ANTENNAS_TX];
  int start_rx_stream = 0;
  AssertFatal(0== openair0_device_load(&(UE->rfdevice), &openair0_cfg[0]), "");
  UE->rfdevice.host_type = RAU_HOST;
  AssertFatal(UE->rfdevice.trx_start_func(&UE->rfdevice) == 0, "Could not start the device\n");
  notifiedFIFO_t nf;
  initNotifiedFIFO(&nf);
  int nbSlotProcessing=0;
  int thread_idx=0;
  notifiedFIFO_t freeBlocks;
  initNotifiedFIFO_nothreadSafe(&freeBlocks);
  NR_UE_MAC_INST_t *mac = get_mac_inst(0);
  int timing_advance = UE->timing_advance;

  for (int i=0; i<RX_NB_TH+1; i++)  // RX_NB_TH working + 1 we are making to be pushed
    pushNotifiedFIFO_nothreadSafe(&freeBlocks,
                                  newNotifiedFIFO_elt(sizeof(processingData_t), 0,&nf,UE_processing));

  bool syncRunning=false;
  const int nb_slot_frame = UE->frame_parms.slots_per_frame;
  int absolute_slot=0, decoded_frame_rx=INT_MAX, trashed_frames=0;

  while (!oai_exit) {
    if (syncRunning) {
      notifiedFIFO_elt_t *res=tryPullTpool(&nf,&(get_nrUE_params()->Tpool));

      if (res) {
        syncRunning=false;
        syncData_t *tmp=(syncData_t *)NotifiedFifoData(res);
        if (UE->is_synchronized) {
          decoded_frame_rx=(((mac->mib->systemFrameNumber.buf[0] >> mac->mib->systemFrameNumber.bits_unused)<<4) | tmp->proc.decoded_frame_rx);
          // shift the frame index with all the frames we trashed meanwhile we perform the synch search
          decoded_frame_rx=(decoded_frame_rx + (!UE->init_sync_frame) + trashed_frames) % MAX_FRAME_NUMBER;
        }
        delNotifiedFIFO_elt(res);
      } else {
        readFrame(UE, &timestamp, true);
        trashed_frames+=2;
        continue;
      }
    }

    AssertFatal( !syncRunning, "At this point synchronization can't be running\n");

    if (!UE->is_synchronized) {
      readFrame(UE, &timestamp, false);
      notifiedFIFO_elt_t *Msg=newNotifiedFIFO_elt(sizeof(syncData_t),0,&nf,UE_synch);
      syncData_t *syncMsg=(syncData_t *)NotifiedFifoData(Msg);
      syncMsg->UE=UE;
      memset(&syncMsg->proc, 0, sizeof(syncMsg->proc));
      pushTpool(&(get_nrUE_params()->Tpool), Msg);
      trashed_frames=0;
      syncRunning=true;
      continue;
    }

    if (start_rx_stream==0) {
      start_rx_stream=1;
      syncInFrame(UE, &timestamp);
      UE->rx_offset=0;
      UE->time_sync_cell=0;
      // read in first symbol
      AssertFatal (UE->frame_parms.ofdm_symbol_size+UE->frame_parms.nb_prefix_samples0 ==
                   UE->rfdevice.trx_read_func(&UE->rfdevice,
                                              &timestamp,
                                              (void **)UE->common_vars.rxdata,
                                              UE->frame_parms.ofdm_symbol_size+UE->frame_parms.nb_prefix_samples0,
                                              UE->frame_parms.nb_antennas_rx),"");
      // we have the decoded frame index in the return of the synch process
      // and we shifted above to the first slot of next frame
      decoded_frame_rx++;
      // we do ++ first in the regular processing, so it will be begin of frame;
      absolute_slot=decoded_frame_rx*nb_slot_frame -1;
      continue;
    }


    absolute_slot++;

    // whatever means thread_idx
    // Fix me: will be wrong when slot 1 is slow, as slot 2 finishes
    // Slot 3 will overlap if RX_NB_TH is 2
    // this is general failure in UE !!!
    thread_idx = absolute_slot % RX_NB_TH;
    int slot_nr = absolute_slot % nb_slot_frame;
    notifiedFIFO_elt_t *msgToPush;
    AssertFatal((msgToPush=pullNotifiedFIFO_nothreadSafe(&freeBlocks)) != NULL,"chained list failure");
    processingData_t *curMsg=(processingData_t *)NotifiedFifoData(msgToPush);
    curMsg->UE=UE;
    // update thread index for received subframe
    curMsg->proc.thread_id   = thread_idx;
    curMsg->proc.CC_id       = UE->CC_id;
    curMsg->proc.nr_slot_rx  = slot_nr;
    curMsg->proc.nr_slot_tx  = (absolute_slot + DURATION_RX_TO_TX) % nb_slot_frame;
    curMsg->proc.frame_rx    = (absolute_slot/nb_slot_frame) % MAX_FRAME_NUMBER;
    curMsg->proc.frame_tx    = ((absolute_slot+DURATION_RX_TO_TX)/nb_slot_frame) % MAX_FRAME_NUMBER;
    curMsg->proc.decoded_frame_rx=-1;
    //LOG_I(PHY,"Process slot %d thread Idx %d total gain %d\n", slot_nr, thread_idx, UE->rx_total_gain_dB);

#ifdef OAI_ADRV9371_ZC706
    /*uint32_t total_gain_dB_prev = 0;
    if (total_gain_dB_prev != UE->rx_total_gain_dB) {
        total_gain_dB_prev = UE->rx_total_gain_dB;
        openair0_cfg[0].rx_gain[0] = UE->rx_total_gain_dB;
        UE->rfdevice.trx_set_gains_func(&UE->rfdevice,&openair0_cfg[0]);
    }*/
#endif

    int firstSymSamp = get_firstSymSamp(slot_nr, &UE->frame_parms);
    for (int i=0; i<UE->frame_parms.nb_antennas_rx; i++)
      rxp[i] = (void *)&UE->common_vars.rxdata[i][firstSymSamp+
               UE->frame_parms.get_samples_slot_timestamp(slot_nr,&UE->frame_parms,0)];

    for (int i=0; i<UE->frame_parms.nb_antennas_tx; i++)
      txp[i] = (void *)&UE->common_vars.txdata[i][UE->frame_parms.get_samples_slot_timestamp(
               ((slot_nr + DURATION_RX_TO_TX - RX_NB_TH)%nb_slot_frame),&UE->frame_parms,0)];

    int readBlockSize, writeBlockSize;

    if (slot_nr<(nb_slot_frame - 1)) {
      readBlockSize=get_readBlockSize(slot_nr, &UE->frame_parms);
      writeBlockSize=UE->frame_parms.get_samples_per_slot((slot_nr + DURATION_RX_TO_TX - RX_NB_TH) % nb_slot_frame, &UE->frame_parms);
    } else {
      UE->rx_offset_diff = computeSamplesShift(UE);
      readBlockSize=get_readBlockSize(slot_nr, &UE->frame_parms) -
                    UE->rx_offset_diff;
      writeBlockSize=UE->frame_parms.get_samples_per_slot((slot_nr + DURATION_RX_TO_TX - RX_NB_TH) % nb_slot_frame, &UE->frame_parms)- UE->rx_offset_diff;
    }

    AssertFatal(readBlockSize ==
                UE->rfdevice.trx_read_func(&UE->rfdevice,
                                           &timestamp,
                                           rxp,
                                           readBlockSize,
                                           UE->frame_parms.nb_antennas_rx),"");

    if( slot_nr==(nb_slot_frame-1)) {
      // read in first symbol of next frame and adjust for timing drift
      int first_symbols=UE->frame_parms.ofdm_symbol_size+UE->frame_parms.nb_prefix_samples0; // first symbol of every frames

      if ( first_symbols > 0 ) {
        openair0_timestamp ignore_timestamp;
        AssertFatal(first_symbols ==
                    UE->rfdevice.trx_read_func(&UE->rfdevice,
                                               &ignore_timestamp,
                                               (void **)UE->common_vars.rxdata,
                                               first_symbols,
                                               UE->frame_parms.nb_antennas_rx),"");
      } else
        LOG_E(PHY,"can't compensate: diff =%d\n", first_symbols);
    }

    curMsg->proc.timestamp_tx = timestamp+
      UE->frame_parms.get_samples_slot_timestamp(slot_nr,&UE->frame_parms,DURATION_RX_TO_TX) 
      - firstSymSamp;

    notifiedFIFO_elt_t *res;

    while (nbSlotProcessing >= RX_NB_TH) {
      res=pullTpool(&nf, &(get_nrUE_params()->Tpool));
      nbSlotProcessing--;
      processingData_t *tmp=(processingData_t *)res->msgData;

      if (tmp->proc.decoded_frame_rx != -1)
        decoded_frame_rx=(((mac->mib->systemFrameNumber.buf[0] >> mac->mib->systemFrameNumber.bits_unused)<<4) | tmp->proc.decoded_frame_rx);
      else 
         decoded_frame_rx=-1;

      pushNotifiedFIFO_nothreadSafe(&freeBlocks,res);
    }

    if (  decoded_frame_rx>0 && decoded_frame_rx != curMsg->proc.frame_rx)
      LOG_E(PHY,"Decoded frame index (%d) is not compatible with current context (%d), UE should go back to synch mode\n",
            decoded_frame_rx, curMsg->proc.frame_rx  );

    // use previous timing_advance value to compute writeTimestamp
    writeTimestamp = timestamp+
      UE->frame_parms.get_samples_slot_timestamp(slot_nr,&UE->frame_parms,DURATION_RX_TO_TX
      - RX_NB_TH) - firstSymSamp - openair0_cfg[0].tx_sample_advance -
      UE->N_TA_offset - timing_advance;

    // but use current UE->timing_advance value to compute writeBlockSize
    if (UE->timing_advance != timing_advance) {
      writeBlockSize -= UE->timing_advance - timing_advance;
      timing_advance = UE->timing_advance;
    }

    int flags = 0;
    int slot_tx_usrp = slot_nr + DURATION_RX_TO_TX - RX_NB_TH;

    if (openair0_cfg[0].duplex_mode == duplex_mode_TDD) {

      uint8_t    tdd_period = mac->phy_config.config_req.tdd_table.tdd_period_in_slots;
      int   nrofUplinkSlots = mac->scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofUplinkSlots;
      uint8_t  num_UL_slots = nrofUplinkSlots + (nrofUplinkSlots != 0);
      uint8_t first_tx_slot = tdd_period - num_UL_slots;

      if (slot_tx_usrp % tdd_period == first_tx_slot)
        flags = 2;
      else if (slot_tx_usrp % tdd_period == first_tx_slot + num_UL_slots - 1)
        flags = 3;
      else if (slot_tx_usrp % tdd_period > first_tx_slot)
        flags = 1;
    } else {
      flags = 1;
    }

    if (flags || IS_SOFTMODEM_RFSIM)
      AssertFatal(writeBlockSize ==
                  UE->rfdevice.trx_write_func(&UE->rfdevice,
                                              writeTimestamp,
                                              txp,
                                              writeBlockSize,
                                              UE->frame_parms.nb_antennas_tx,
                                              flags),"");
    
    for (int i=0; i<UE->frame_parms.nb_antennas_tx; i++)
      memset(txp[i], 0, writeBlockSize);

    nbSlotProcessing++;
    msgToPush->key=slot_nr;
    pushTpool(&(get_nrUE_params()->Tpool), msgToPush);

    if (IS_SOFTMODEM_RFSIM) {  //getenv("RFSIMULATOR")
      // FixMe: Wait previous thread is done, because race conditions seems too bad
      // in case of actual RF board, the overlap between threads mitigate the issue
      // We must receive one message, that proves the slot processing is done
      res=pullTpool(&nf, &(get_nrUE_params()->Tpool));
      nbSlotProcessing--;
      processingData_t *tmp=(processingData_t *)res->msgData;

      if (tmp->proc.decoded_frame_rx != -1)
        decoded_frame_rx=(((mac->mib->systemFrameNumber.buf[0] >> mac->mib->systemFrameNumber.bits_unused)<<4) | tmp->proc.decoded_frame_rx);
      else 
        decoded_frame_rx=-1;
        //decoded_frame_rx=tmp->proc.decoded_frame_rx;

      pushNotifiedFIFO_nothreadSafe(&freeBlocks,res);
    }

  } // while !oai_exit

  return NULL;
}

void init_NR_UE(int nb_inst, char* rrc_config_path) {
  int inst;
  NR_UE_MAC_INST_t *mac_inst;
  NR_UE_RRC_INST_t* rrc_inst;
  
  for (inst=0; inst < nb_inst; inst++) {
    AssertFatal((rrc_inst = nr_l3_init_ue(rrc_config_path)) != NULL, "can not initialize RRC module\n");
    AssertFatal((mac_inst = nr_l2_init_ue(rrc_inst)) != NULL, "can not initialize L2 module\n");
    AssertFatal((mac_inst->if_module = nr_ue_if_module_init(inst)) != NULL, "can not initialize IF module\n");
  }
}

void init_NR_UE_threads(int nb_inst) {
  int inst;

  pthread_t threads[nb_inst];

  for (inst=0; inst < nb_inst; inst++) {
    PHY_VARS_NR_UE *UE = PHY_vars_UE_g[inst][0];

    LOG_I(PHY,"Intializing UE Threads for instance %d (%p,%p)...\n",inst,PHY_vars_UE_g[inst],PHY_vars_UE_g[inst][0]);
    threadCreate(&threads[inst], UE_thread, (void *)UE, "UEthread", -1, OAI_PRIORITY_RT_MAX);

     if(get_nrUE_params()->nr_dlsch_parallel)
     {
       pthread_t dlsch0_threads;
       threadCreate(&dlsch0_threads, dlsch_thread, (void *)UE, "DLthread", -1, OAI_PRIORITY_RT_MAX-1);
     }
  }
}

/* HACK: this function is needed to compile the UE
 * fix it somehow
 */
int8_t find_dlsch(uint16_t rnti,
                  PHY_VARS_eNB *eNB,
                  find_type_t type)
{
  printf("you cannot read this\n");
  abort();
}

void multicast_link_write_sock(int groupP, char *dataP, uint32_t sizeP) {}
