/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
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

#include "PHY/phy_extern.h"
#include "PHY/defs_gNB.h"
#include "sched_nr.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "PHY/NR_TRANSPORT/nr_ulsch.h"
#include "PHY/NR_TRANSPORT/nr_dci.h"
#include "PHY/NR_ESTIMATION/nr_ul_estimation.h"
#include "PHY/NR_UE_TRANSPORT/pucch_nr.h"
#include "SCHED/sched_eNB.h"
#include "sched_nr.h"
#include "SCHED/sched_common_extern.h"
#include "nfapi/open-nFAPI/nfapi/public_inc/nfapi_interface.h"
#include "nfapi/open-nFAPI/nfapi/public_inc/nfapi_nr_interface.h"
#include "fapi_nr_l1.h"
#include "common/utils/LOG/log.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "PHY/INIT/phy_init.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "T.h"
#include "executables/nr-softmodem.h"
#include "executables/softmodem-common.h"

#include "assertions.h"
#include "msc.h"

#include <time.h>

#include "intertask_interface.h"

//#define DEBUG_RXDATA

uint8_t SSB_Table[38]={0,2,4,6,8,10,12,14,254,254,16,18,20,22,24,26,28,30,254,254,32,34,36,38,40,42,44,46,254,254,48,50,52,54,56,58,60,62};

extern uint8_t nfapi_mode;

void nr_set_ssb_first_subcarrier(nfapi_nr_config_request_scf_t *cfg, NR_DL_FRAME_PARMS *fp) {

  uint8_t sco = 0;
  if (((fp->freq_range == nr_FR1) && (cfg->ssb_table.ssb_subcarrier_offset.value<24)) ||
      ((fp->freq_range == nr_FR2) && (cfg->ssb_table.ssb_subcarrier_offset.value<12)) )
    sco = cfg->ssb_table.ssb_subcarrier_offset.value;

  fp->ssb_start_subcarrier = (12 * cfg->ssb_table.ssb_offset_point_a.value + sco);
  LOG_D(PHY, "SSB first subcarrier %d (%d,%d)\n", fp->ssb_start_subcarrier,cfg->ssb_table.ssb_offset_point_a.value,sco);
}

void nr_common_signal_procedures (PHY_VARS_gNB *gNB,int frame, int slot) {

  NR_DL_FRAME_PARMS *fp=&gNB->frame_parms;
  nfapi_nr_config_request_scf_t *cfg = &gNB->gNB_config;
  int **txdataF = gNB->common_vars.txdataF;
  uint8_t ssb_index, n_hf;
  uint16_t ssb_start_symbol, rel_slot;
  int txdataF_offset = (slot%2)*fp->samples_per_slot_wCP;
  uint16_t slots_per_hf = (fp->slots_per_frame)>>1;

  n_hf = fp->half_frame_bit;

  // if SSB periodicity is 5ms, they are transmitted in both half frames
  if ( cfg->ssb_table.ssb_period.value == 0) {
    if (slot<slots_per_hf)
      n_hf=0;
    else
      n_hf=1;
  }

  // to set a effective slot number in the half frame where the SSB is supposed to be
  rel_slot = (n_hf)? (slot-slots_per_hf) : slot; 

  LOG_D(PHY,"common_signal_procedures: frame %d, slot %d\n",frame,slot);

  if(rel_slot<38 && rel_slot>=0)  { // there is no SSB beyond slot 37

    for (int i=0; i<2; i++)  {  // max two SSB per frame
      
      ssb_index = i + SSB_Table[rel_slot]; // computing the ssb_index

      if ((ssb_index<64) && ((fp->L_ssb >> (63-ssb_index)) & 0x01))  { // generating the ssb only if the bit of L_ssb at current ssb index is 1
        fp->ssb_index = ssb_index;
        int ssb_start_symbol_abs = nr_get_ssb_start_symbol(fp); // computing the starting symbol for current ssb
	ssb_start_symbol = ssb_start_symbol_abs % fp->symbols_per_slot;  // start symbol wrt slot

	nr_set_ssb_first_subcarrier(cfg, fp);  // setting the first subcarrier
	
	LOG_D(PHY,"SS TX: frame %d, slot %d, start_symbol %d\n",frame,slot, ssb_start_symbol);
	nr_generate_pss(gNB->d_pss, &txdataF[0][txdataF_offset], AMP, ssb_start_symbol, cfg, fp);
	nr_generate_sss(gNB->d_sss, &txdataF[0][txdataF_offset], AMP, ssb_start_symbol, cfg, fp);
	
        if (cfg->carrier_config.num_tx_ant.value <= 4)
	  nr_generate_pbch_dmrs(gNB->nr_gold_pbch_dmrs[n_hf][ssb_index&7],&txdataF[0][txdataF_offset], AMP, ssb_start_symbol, cfg, fp);
        else
	  nr_generate_pbch_dmrs(gNB->nr_gold_pbch_dmrs[0][ssb_index&7],&txdataF[0][txdataF_offset], AMP, ssb_start_symbol, cfg, fp);

        if (T_ACTIVE(T_GNB_PHY_MIB)) {
          unsigned char bch[3];
          bch[0] = gNB->ssb_pdu.ssb_pdu_rel15.bchPayload & 0xff;
          bch[1] = (gNB->ssb_pdu.ssb_pdu_rel15.bchPayload >> 8) & 0xff;
          bch[2] = (gNB->ssb_pdu.ssb_pdu_rel15.bchPayload >> 16) & 0xff;
          T(T_GNB_PHY_MIB, T_INT(0) /* module ID */, T_INT(frame), T_INT(slot), T_BUFFER(bch, 3));
        }

	nr_generate_pbch(&gNB->pbch,
	                 &gNB->ssb_pdu,
	                 gNB->nr_pbch_interleaver,
			 &txdataF[0][txdataF_offset],
			 AMP,
			 ssb_start_symbol,
			 n_hf, frame, cfg, fp);

      }
    }
  }
}

void phy_procedures_gNB_TX(PHY_VARS_gNB *gNB,
                           int frame,int slot,
                           int do_meas) {
  int aa;
  NR_DL_FRAME_PARMS *fp=&gNB->frame_parms;
  nfapi_nr_config_request_scf_t *cfg = &gNB->gNB_config;
  int offset = gNB->CC_id;
  uint8_t ssb_frame_periodicity = 1;  // every how many frames SSB are generated
  int txdataF_offset = (slot%2)*fp->samples_per_slot_wCP;
  
  if (cfg->ssb_table.ssb_period.value > 1) 
    ssb_frame_periodicity = 1 <<(cfg->ssb_table.ssb_period.value -1) ; 

  if ((cfg->cell_config.frame_duplex_type.value == TDD) &&
      (nr_slot_select(cfg,frame,slot) == NR_UPLINK_SLOT)) return;

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_gNB_TX+offset,1);

  if (do_meas==1) start_meas(&gNB->phy_proc_tx);

  // clear the transmit data array for the current subframe
  for (aa=0; aa<cfg->carrier_config.num_tx_ant.value; aa++) {
    memset(&gNB->common_vars.txdataF[aa][txdataF_offset],0,fp->samples_per_slot_wCP*sizeof(int32_t));
  }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_gNB_COMMON_TX,1);
  if (NFAPI_MODE == NFAPI_MONOLITHIC || NFAPI_MODE == NFAPI_MODE_PNF) { 
    if ((!(frame%ssb_frame_periodicity)))  // generate SSB only for given frames according to SSB periodicity
      nr_common_signal_procedures(gNB,frame, slot);
  }
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_gNB_COMMON_TX,0);

  int pdcch_pdu_id=find_nr_pdcch(frame,slot,gNB,SEARCH_EXIST);
  int ul_pdcch_pdu_id=find_nr_ul_dci(frame,slot,gNB,SEARCH_EXIST);

  LOG_D(PHY,"[gNB %d] Frame %d slot %d, pdcch_pdu_id %d, ul_pdcch_pdu_id %d\n",
	gNB->Mod_id,frame,slot,pdcch_pdu_id,ul_pdcch_pdu_id);

  if (pdcch_pdu_id >= 0 || ul_pdcch_pdu_id >= 0) {
    LOG_D(PHY, "[gNB %d] Frame %d slot %d Calling nr_generate_dci_top (number of UL/DL DCI %d/%d)\n",
	  gNB->Mod_id, frame, slot,
	  gNB->ul_pdcch_pdu[ul_pdcch_pdu_id].pdcch_pdu.pdcch_pdu.pdcch_pdu_rel15.numDlDci,
	  gNB->pdcch_pdu[pdcch_pdu_id].pdcch_pdu.pdcch_pdu_rel15.numDlDci);
  
    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_gNB_PDCCH_TX,1);

    nr_generate_dci_top(gNB,
			pdcch_pdu_id>=0 ? &gNB->pdcch_pdu[pdcch_pdu_id].pdcch_pdu : NULL,
			ul_pdcch_pdu_id>=0 ? &gNB->ul_pdcch_pdu[ul_pdcch_pdu_id].pdcch_pdu.pdcch_pdu : NULL,
			gNB->nr_gold_pdcch_dmrs[slot],
			&gNB->common_vars.txdataF[0][txdataF_offset],
			AMP, *fp);

    // free up entry in pdcch tables
    if (pdcch_pdu_id>=0) gNB->pdcch_pdu[pdcch_pdu_id].frame = -1;
    if (ul_pdcch_pdu_id>=0) gNB->ul_pdcch_pdu[ul_pdcch_pdu_id].frame = -1;

    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_gNB_PDCCH_TX,0);
    if (pdcch_pdu_id >= 0) gNB->pdcch_pdu[pdcch_pdu_id].frame = -1;
    if (ul_pdcch_pdu_id >= 0) gNB->ul_pdcch_pdu[ul_pdcch_pdu_id].frame = -1;
  }
 
  for (int i=0; i<gNB->num_pdsch_rnti[slot]; i++) {
    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_GENERATE_DLSCH,1);
    LOG_D(PHY, "PDSCH generation started (%d) in frame %d.%d\n", gNB->num_pdsch_rnti[slot],frame,slot);
    nr_generate_pdsch(gNB,frame, slot);
    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_GENERATE_DLSCH,0);
  }

  if ((frame&127) == 0) dump_pdsch_stats(gNB);

  //apply the OFDM symbol rotation here
  apply_nr_rotation(fp,(int16_t*) &gNB->common_vars.txdataF[0][txdataF_offset],slot,0,fp->Ncp==EXTENDED?12:14,fp->ofdm_symbol_size);
  
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_gNB_TX+offset,0);
}



/*

  if ((cfg->subframe_config.duplex_mode.value == TDD) && 
      ((nr_slot_select(fp,frame,slot)&NR_DOWNLINK_SLOT)==SF_DL)) return;

  //  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_ENB_RX,1);

*/

void nr_postDecode(PHY_VARS_gNB *gNB, notifiedFIFO_elt_t *req) {
  ldpcDecode_t *rdata = (ldpcDecode_t*) NotifiedFifoData(req);
  NR_UL_gNB_HARQ_t *ulsch_harq = rdata->ulsch_harq;
  NR_gNB_ULSCH_t *ulsch = rdata->ulsch;
  int r = rdata->segment_r;

  bool decodeSuccess = (rdata->decodeIterations <= rdata->decoderParms.numMaxIter);
  ulsch_harq->processedSegments++;
  LOG_D(PHY, "processing result of segment: %d, processed %d/%d\n",
	rdata->segment_r, ulsch_harq->processedSegments, rdata->nbSegments);
  gNB->nbDecode--;
  LOG_D(PHY,"remain to decoded in subframe: %d\n", gNB->nbDecode);
  
  if (decodeSuccess) {
    memcpy(ulsch_harq->b+rdata->offset,
           ulsch_harq->c[r],
           rdata->Kr_bytes - (ulsch_harq->F>>3) -((ulsch_harq->C>1)?3:0));

  } else {
    if ( rdata->nbSegments != ulsch_harq->processedSegments ) {
      int nb=abortTpool(gNB->threadPool, req->key);
      nb+=abortNotifiedFIFO(gNB->respDecode, req->key);
      gNB->nbDecode-=nb;
      LOG_D(PHY,"uplink segment error %d/%d, aborted %d segments\n",rdata->segment_r,rdata->nbSegments, nb);
      LOG_D(PHY, "ULSCH %d in error\n",rdata->ulsch_id);
      AssertFatal(ulsch_harq->processedSegments+nb == rdata->nbSegments,"processed: %d, aborted: %d, total %d\n",
		  ulsch_harq->processedSegments, nb, rdata->nbSegments);
      ulsch_harq->processedSegments=rdata->nbSegments;
    }
  }

  // if all segments are done 
  if (rdata->nbSegments == ulsch_harq->processedSegments) {
    if (decodeSuccess) {
      LOG_D(PHY,"[gNB %d] ULSCH: Setting ACK for slot %d TBS %d\n",
            gNB->Mod_id,ulsch_harq->slot,ulsch_harq->TBS);
      ulsch_harq->status = SCH_IDLE;
      ulsch_harq->round  = 0;
      ulsch->harq_mask &= ~(1 << rdata->harq_pid);

      LOG_D(PHY, "ULSCH received ok \n");
      nr_fill_indication(gNB,ulsch_harq->frame, ulsch_harq->slot, rdata->ulsch_id, rdata->harq_pid, 0);
    } else {
      LOG_D(PHY,"[gNB %d] ULSCH: Setting NAK for SFN/SF %d/%d (pid %d, status %d, round %d, TBS %d) r %d\n",
            gNB->Mod_id, ulsch_harq->frame, ulsch_harq->slot,
            rdata->harq_pid,ulsch_harq->status, ulsch_harq->round,ulsch_harq->TBS,r);
      if (ulsch_harq->round >= ulsch->Mlimit) {
        ulsch_harq->status = SCH_IDLE;
        ulsch_harq->round  = 0;
        ulsch_harq->handled  = 0;
        ulsch->harq_mask &= ~(1 << rdata->harq_pid);
      }
      ulsch_harq->handled  = 1;

      LOG_D(PHY, "ULSCH %d in error\n",rdata->ulsch_id);
      nr_fill_indication(gNB,ulsch_harq->frame, ulsch_harq->slot, rdata->ulsch_id, rdata->harq_pid, 1);
    }
    ulsch->last_iteration_cnt = rdata->decodeIterations;
    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_gNB_ULSCH_DECODING,0);
  }
}


void nr_ulsch_procedures(PHY_VARS_gNB *gNB, int frame_rx, int slot_rx, int ULSCH_id, uint8_t harq_pid)
{
  NR_DL_FRAME_PARMS *frame_parms = &gNB->frame_parms;
  nfapi_nr_pusch_pdu_t *pusch_pdu = &gNB->ulsch[ULSCH_id][0]->harq_processes[harq_pid]->ulsch_pdu;
  
  uint8_t l, number_dmrs_symbols = 0;
  uint32_t G;
  uint16_t start_symbol, number_symbols, nb_re_dmrs;

  start_symbol = pusch_pdu->start_symbol_index;
  number_symbols = pusch_pdu->nr_of_symbols;

  for (l = start_symbol; l < start_symbol + number_symbols; l++)
    number_dmrs_symbols += ((pusch_pdu->ul_dmrs_symb_pos)>>l)&0x01;

  if (pusch_pdu->dmrs_config_type==pusch_dmrs_type1)
    nb_re_dmrs = 6*pusch_pdu->num_dmrs_cdm_grps_no_data;
  else
    nb_re_dmrs = 4*pusch_pdu->num_dmrs_cdm_grps_no_data;

  G = nr_get_G(pusch_pdu->rb_size,
               number_symbols,
               nb_re_dmrs,
               number_dmrs_symbols, // number of dmrs symbols irrespective of single or double symbol dmrs
               pusch_pdu->qam_mod_order,
               pusch_pdu->nrOfLayers);
  
  AssertFatal(G>0,"G is 0 : rb_size %u, number_symbols %d, nb_re_dmrs %d, number_dmrs_symbols %d, qam_mod_order %u, nrOfLayer %u\n",
	      pusch_pdu->rb_size,
	      number_symbols,
	      nb_re_dmrs,
	      number_dmrs_symbols, // number of dmrs symbols irrespective of single or double symbol dmrs
	      pusch_pdu->qam_mod_order,
	      pusch_pdu->nrOfLayers);
  LOG_D(PHY,"rb_size %d, number_symbols %d, nb_re_dmrs %d, number_dmrs_symbols %d, qam_mod_order %d, nrOfLayer %d\n",
	pusch_pdu->rb_size,
	number_symbols,
	nb_re_dmrs,
	number_dmrs_symbols, // number of dmrs symbols irrespective of single or double symbol dmrs
	pusch_pdu->qam_mod_order,
	pusch_pdu->nrOfLayers);
  //----------------------------------------------------------
  //------------------- ULSCH unscrambling -------------------
  //----------------------------------------------------------
  start_meas(&gNB->ulsch_unscrambling_stats);
  nr_ulsch_unscrambling_optim(gNB->pusch_vars[ULSCH_id]->llr,
			      G,
			      0,
			      pusch_pdu->data_scrambling_id,
			      pusch_pdu->rnti);
  stop_meas(&gNB->ulsch_unscrambling_stats);
  //----------------------------------------------------------
  //--------------------- ULSCH decoding ---------------------
  //----------------------------------------------------------

  start_meas(&gNB->ulsch_decoding_stats);
  nr_ulsch_decoding(gNB,
                    ULSCH_id,
                    gNB->pusch_vars[ULSCH_id]->llr,
                    frame_parms,
                    pusch_pdu,
                    frame_rx,
                    slot_rx,
                    harq_pid,
                    G);

  while (gNB->nbDecode > 0) {
    notifiedFIFO_elt_t *req=pullTpool(gNB->respDecode, gNB->threadPool);
    nr_postDecode(gNB, req);
    delNotifiedFIFO_elt(req);
  }
  stop_meas(&gNB->ulsch_decoding_stats);
}


void nr_fill_indication(PHY_VARS_gNB *gNB, int frame, int slot_rx, int ULSCH_id, uint8_t harq_pid, uint8_t crc_flag) {

  pthread_mutex_lock(&gNB->UL_INFO_mutex);

  int timing_advance_update, cqi;
  int sync_pos;
  uint16_t mu = gNB->frame_parms.numerology_index;
  NR_gNB_ULSCH_t                       *ulsch                 = gNB->ulsch[ULSCH_id][0];
  NR_UL_gNB_HARQ_t                     *harq_process          = ulsch->harq_processes[harq_pid];

  nfapi_nr_pusch_pdu_t *pusch_pdu = &harq_process->ulsch_pdu;

  //  pdu->data                              = gNB->ulsch[ULSCH_id+1][0]->harq_processes[harq_pid]->b;
  sync_pos                               = nr_est_timing_advance_pusch(gNB, ULSCH_id); // estimate timing advance for MAC
  timing_advance_update                  = sync_pos * (1 << mu);                    // scale by the used scs numerology

  // scale the 16 factor in N_TA calculation in 38.213 section 4.2 according to the used FFT size
  switch (gNB->frame_parms.N_RB_DL) {
    case 106: timing_advance_update /= 16; break;
    case 217: timing_advance_update /= 32; break;
    case 245: timing_advance_update /= 32; break;
    case 273: timing_advance_update /= 32; break;
    case 66:  timing_advance_update /= 12; break;
    case 32:  timing_advance_update /= 12; break;
    default: AssertFatal(0==1,"No case defined for PRB %d to calculate timing_advance_update\n",gNB->frame_parms.N_RB_DL);
  }

  // put timing advance command in 0..63 range
  timing_advance_update += 31;

  if (timing_advance_update < 0)  timing_advance_update = 0;
  if (timing_advance_update > 63) timing_advance_update = 63;

  LOG_D(PHY, "Estimated timing advance PUSCH is  = %d, timing_advance_update is %d \n", sync_pos,timing_advance_update);

  // estimate UL_CQI for MAC (from antenna port 0 only)
  int SNRtimes10 = dB_fixed_times10(gNB->pusch_vars[ULSCH_id]->ulsch_power[0]) - (10*gNB->measurements.n0_power_dB[0]);

  LOG_D(PHY, "Estimated SNR for PUSCH is = %d dB\n", SNRtimes10/10);

  if      (SNRtimes10 < -640) cqi=0;
  else if (SNRtimes10 >  635) cqi=255;
  else                        cqi=(640+SNRtimes10)/5;

  // crc indication
  uint16_t num_crc = gNB->UL_INFO.crc_ind.number_crcs;
  gNB->UL_INFO.crc_ind.crc_list = &gNB->crc_pdu_list[0];
  gNB->UL_INFO.crc_ind.sfn = frame;
  gNB->UL_INFO.crc_ind.slot = slot_rx;

  gNB->crc_pdu_list[num_crc].handle = pusch_pdu->handle;
  gNB->crc_pdu_list[num_crc].rnti = pusch_pdu->rnti;
  gNB->crc_pdu_list[num_crc].harq_id = harq_pid;
  gNB->crc_pdu_list[num_crc].tb_crc_status = crc_flag;
  gNB->crc_pdu_list[num_crc].num_cb = pusch_pdu->pusch_data.num_cb;
  gNB->crc_pdu_list[num_crc].ul_cqi = cqi;
  gNB->crc_pdu_list[num_crc].timing_advance = timing_advance_update;
  // in terms of dBFS range -128 to 0 with 0.1 step
  gNB->crc_pdu_list[num_crc].rssi = 1280 - (10*dB_fixed(32767*32767)-dB_fixed_times10(gNB->pusch_vars[ULSCH_id]->ulsch_power[0]));

  gNB->UL_INFO.crc_ind.number_crcs++;

  // rx indication
  uint16_t num_rx = gNB->UL_INFO.rx_ind.number_of_pdus;
  gNB->UL_INFO.rx_ind.pdu_list = &gNB->rx_pdu_list[0];
  gNB->UL_INFO.rx_ind.sfn = frame;
  gNB->UL_INFO.rx_ind.slot = slot_rx;
  gNB->rx_pdu_list[num_rx].handle = pusch_pdu->handle;
  gNB->rx_pdu_list[num_rx].rnti = pusch_pdu->rnti;
  gNB->rx_pdu_list[num_rx].harq_id = harq_pid;
  gNB->rx_pdu_list[num_rx].ul_cqi = cqi;
  gNB->rx_pdu_list[num_rx].timing_advance = timing_advance_update;
  gNB->rx_pdu_list[num_rx].rssi = 1280 - (10*dB_fixed(32767*32767)-dB_fixed_times10(gNB->pusch_vars[ULSCH_id]->ulsch_power[0]));
  if (crc_flag)
    gNB->rx_pdu_list[num_rx].pdu_length = 0;
  else {
    gNB->rx_pdu_list[num_rx].pdu_length = harq_process->TBS;
    gNB->rx_pdu_list[num_rx].pdu = harq_process->b;
  }

  gNB->UL_INFO.rx_ind.number_of_pdus++;

  pthread_mutex_unlock(&gNB->UL_INFO_mutex);
}

// Function to fill UL RB mask to be used for N0 measurements
void fill_ul_rb_mask(PHY_VARS_gNB *gNB, int frame_rx, int slot_rx) {

  int rb2, rb, nb_rb;
  for (int symbol=0;symbol<14;symbol++) {
    if (gNB->gNB_config.tdd_table.max_tdd_periodicity_list[slot_rx].max_num_of_symbol_per_slot_list[symbol].slot_config.value==1){
      nb_rb = 0;
      for (int m=0;m<9;m++) gNB->rb_mask_ul[m] = 0;
      gNB->ulmask_symb = -1;

      for (int i=0;i<NUMBER_OF_NR_PUCCH_MAX;i++){
        NR_gNB_PUCCH_t *pucch = gNB->pucch[i];
        if (pucch) {
          if ((pucch->active == 1) &&
	      (pucch->frame == frame_rx) &&
	      (pucch->slot == slot_rx) ) {
            gNB->ulmask_symb = symbol;
            nfapi_nr_pucch_pdu_t  *pucch_pdu = &pucch[i].pucch_pdu;
            if ((symbol>=pucch_pdu->start_symbol_index) &&
                (symbol<(pucch_pdu->start_symbol_index + pucch_pdu->nr_of_symbols))){
              for (rb=0; rb<pucch_pdu->prb_size; rb++) {
                rb2 = rb+pucch_pdu->prb_start;
                gNB->rb_mask_ul[rb2>>5] |= (1<<(rb2&31));
              }
              nb_rb+=pucch_pdu->prb_size;
            }
          }
        }
      }
      for (int ULSCH_id=0;ULSCH_id<NUMBER_OF_NR_ULSCH_MAX;ULSCH_id++) {
        NR_gNB_ULSCH_t *ulsch = gNB->ulsch[ULSCH_id][0];
        int harq_pid;
        NR_UL_gNB_HARQ_t *ulsch_harq;

        if ((ulsch) &&
            (ulsch->rnti > 0)) {
          for (harq_pid=0;harq_pid<NR_MAX_ULSCH_HARQ_PROCESSES;harq_pid++) {
            ulsch_harq = ulsch->harq_processes[harq_pid];
            AssertFatal(ulsch_harq!=NULL,"harq_pid %d is not allocated\n",harq_pid);
            if ((ulsch_harq->status == NR_ACTIVE) &&
                (ulsch_harq->frame == frame_rx) &&
                (ulsch_harq->slot == slot_rx) &&
                (ulsch_harq->handled == 0)){
              uint8_t symbol_start = ulsch_harq->ulsch_pdu.start_symbol_index;
              uint8_t symbol_end = symbol_start + ulsch_harq->ulsch_pdu.nr_of_symbols;
              gNB->ulmask_symb = symbol;
              if ((symbol>=symbol_start) &&
                  (symbol<symbol_end)){
                for (rb=0; rb<ulsch_harq->ulsch_pdu.rb_size; rb++) {
                  rb2 = rb+ulsch_harq->ulsch_pdu.rb_start;
                  gNB->rb_mask_ul[rb2>>5] |= (1<<(rb2&31));
                }
                nb_rb+=ulsch_harq->ulsch_pdu.rb_size;
              }
            }
          }
        }
      //TODO Add check for PRACH as well?
      }
      if (nb_rb<gNB->frame_parms.N_RB_UL)
        return;
    }
  }
}

void phy_procedures_gNB_common_RX(PHY_VARS_gNB *gNB, int frame_rx, int slot_rx) {

  uint8_t symbol;
  unsigned char aa;

  for(symbol = 0; symbol < (gNB->frame_parms.Ncp==EXTENDED?12:14); symbol++) {
    // nr_slot_fep_ul(gNB, symbol, proc->slot_rx, 0, 0);

    for (aa = 0; aa < gNB->frame_parms.nb_antennas_rx; aa++) {
      nr_slot_fep_ul(&gNB->frame_parms,
                     gNB->common_vars.rxdata[aa],
                     gNB->common_vars.rxdataF[aa],
                     symbol,
                     slot_rx,
                     0,
                     0);
    }
  }

  for (aa = 0; aa < gNB->frame_parms.nb_antennas_rx; aa++) {
    apply_nr_rotation_ul(&gNB->frame_parms,
			 gNB->common_vars.rxdataF[aa],
			 slot_rx,
			 0,
			 gNB->frame_parms.Ncp==EXTENDED?12:14,
			 gNB->frame_parms.ofdm_symbol_size);
  }

}

void phy_procedures_gNB_uespec_RX(PHY_VARS_gNB *gNB, int frame_rx, int slot_rx) {

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_gNB_UESPEC_RX,1);
  LOG_D(PHY,"phy_procedures_gNB_uespec_RX frame %d, slot %d\n",frame_rx,slot_rx);

  if (gNB->frame_parms.frame_type == TDD)
    fill_ul_rb_mask(gNB, frame_rx, slot_rx);

  gNB_I0_measurements(gNB);

  // measure enegry in SS=10 L=4, nb_rb = 18, first_rb = 0 (corresponds to msg3)
  int offset = 10*gNB->frame_parms.ofdm_symbol_size + gNB->frame_parms.first_carrier_offset;
  int power_rxF = signal_energy_nodc(&gNB->common_vars.rxdataF[0][offset],12*18);
  LOG_D(PHY,"frame %d, slot %d: UL signal energy %d\n",frame_rx,slot_rx,power_rxF);

  for (int i=0;i<NUMBER_OF_NR_PUCCH_MAX;i++){
    NR_gNB_PUCCH_t *pucch = gNB->pucch[i];
    if (pucch) {
      if ((pucch->active == 1) &&
	  (pucch->frame == frame_rx) &&
	  (pucch->slot == slot_rx) ) {

        nfapi_nr_pucch_pdu_t  *pucch_pdu = &pucch->pucch_pdu;
        uint16_t num_ucis;
        switch (pucch_pdu->format_type) {
        case 0:
          num_ucis = gNB->UL_INFO.uci_ind.num_ucis;
          gNB->UL_INFO.uci_ind.uci_list = &gNB->uci_pdu_list[0];
          gNB->UL_INFO.uci_ind.sfn = frame_rx;
          gNB->UL_INFO.uci_ind.slot = slot_rx;
          gNB->uci_pdu_list[num_ucis].pdu_type = NFAPI_NR_UCI_FORMAT_0_1_PDU_TYPE;
          gNB->uci_pdu_list[num_ucis].pdu_size = sizeof(nfapi_nr_uci_pucch_pdu_format_0_1_t);
          nfapi_nr_uci_pucch_pdu_format_0_1_t *uci_pdu_format0 = &gNB->uci_pdu_list[num_ucis].pucch_pdu_format_0_1;

          nr_decode_pucch0(gNB,
	                   slot_rx,
                           uci_pdu_format0,
                           pucch_pdu);

          gNB->UL_INFO.uci_ind.num_ucis += 1;
          pucch->active = 0;
	  break;
        case 2:
          num_ucis = gNB->UL_INFO.uci_ind.num_ucis;
          gNB->UL_INFO.uci_ind.uci_list = &gNB->uci_pdu_list[0];
          gNB->UL_INFO.uci_ind.sfn = frame_rx;
          gNB->UL_INFO.uci_ind.slot = slot_rx;
          gNB->uci_pdu_list[num_ucis].pdu_type = NFAPI_NR_UCI_FORMAT_2_3_4_PDU_TYPE;
          gNB->uci_pdu_list[num_ucis].pdu_size = sizeof(nfapi_nr_uci_pucch_pdu_format_2_3_4_t);
          nfapi_nr_uci_pucch_pdu_format_2_3_4_t *uci_pdu_format2 = &gNB->uci_pdu_list[num_ucis].pucch_pdu_format_2_3_4;

          nr_decode_pucch2(gNB,
                           slot_rx,
                           uci_pdu_format2,
                           pucch_pdu);

          gNB->UL_INFO.uci_ind.num_ucis += 1;
          pucch->active = 0;
          break;
        default:
	  AssertFatal(1==0,"Only PUCCH formats 0 and 2 are currently supported\n");
        }
      }
    }
  }

  for (int ULSCH_id=0;ULSCH_id<NUMBER_OF_NR_ULSCH_MAX;ULSCH_id++) {
    NR_gNB_ULSCH_t *ulsch = gNB->ulsch[ULSCH_id][0];
    int harq_pid;
    int no_sig;
    NR_UL_gNB_HARQ_t *ulsch_harq;

    if ((ulsch) &&
        (ulsch->rnti > 0)) {
      // for for an active HARQ process
      for (harq_pid=0;harq_pid<NR_MAX_ULSCH_HARQ_PROCESSES;harq_pid++) {
	ulsch_harq = ulsch->harq_processes[harq_pid];
    	AssertFatal(ulsch_harq!=NULL,"harq_pid %d is not allocated\n",harq_pid);
    	if ((ulsch_harq->status == NR_ACTIVE) &&
          (ulsch_harq->frame == frame_rx) &&
          (ulsch_harq->slot == slot_rx) &&
          (ulsch_harq->handled == 0)){

          LOG_D(PHY, "PUSCH detection started in frame %d slot %d\n",
                frame_rx,slot_rx);

#ifdef DEBUG_RXDATA
          NR_DL_FRAME_PARMS *frame_parms = &gNB->frame_parms;
          RU_t *ru = gNB->RU_list[0];
          int slot_offset = frame_parms->get_samples_slot_timestamp(slot_rx,frame_parms,0);
          slot_offset -= ru->N_TA_offset;
          ((int16_t*)&gNB->common_vars.debugBuff[gNB->common_vars.debugBuff_sample_offset])[0]=(int16_t)ulsch->rnti;
          ((int16_t*)&gNB->common_vars.debugBuff[gNB->common_vars.debugBuff_sample_offset])[1]=(int16_t)ulsch_harq->ulsch_pdu.rb_size;
          ((int16_t*)&gNB->common_vars.debugBuff[gNB->common_vars.debugBuff_sample_offset])[2]=(int16_t)ulsch_harq->ulsch_pdu.rb_start;
          ((int16_t*)&gNB->common_vars.debugBuff[gNB->common_vars.debugBuff_sample_offset])[3]=(int16_t)ulsch_harq->ulsch_pdu.nr_of_symbols;
          ((int16_t*)&gNB->common_vars.debugBuff[gNB->common_vars.debugBuff_sample_offset])[4]=(int16_t)ulsch_harq->ulsch_pdu.start_symbol_index;
          ((int16_t*)&gNB->common_vars.debugBuff[gNB->common_vars.debugBuff_sample_offset])[5]=(int16_t)ulsch_harq->ulsch_pdu.mcs_index;
          ((int16_t*)&gNB->common_vars.debugBuff[gNB->common_vars.debugBuff_sample_offset])[6]=(int16_t)ulsch_harq->ulsch_pdu.pusch_data.rv_index;
          ((int16_t*)&gNB->common_vars.debugBuff[gNB->common_vars.debugBuff_sample_offset])[7]=(int16_t)harq_pid;
          memcpy(&gNB->common_vars.debugBuff[gNB->common_vars.debugBuff_sample_offset+4],&ru->common.rxdata[0][slot_offset],frame_parms->get_samples_per_slot(slot_rx,frame_parms)*sizeof(int32_t));
          gNB->common_vars.debugBuff_sample_offset+=(frame_parms->get_samples_per_slot(slot_rx,frame_parms)+1000+4);
          if(gNB->common_vars.debugBuff_sample_offset>((frame_parms->get_samples_per_slot(slot_rx,frame_parms)+1000+2)*20)) {
            FILE *f;
            f = fopen("rxdata_buff.raw", "w"); if (f == NULL) exit(1);
            fwrite((int16_t*)gNB->common_vars.debugBuff,2,(frame_parms->get_samples_per_slot(slot_rx,frame_parms)+1000+4)*20*2, f);
            fclose(f);
            exit(-1);
          }
#endif

T(T_BENETEL, T_INT(frame_rx), T_INT(slot_rx), T_BUFFER(&gNB->common_vars.rxdataF[0][0], 2048*4*14));

          uint8_t symbol_start = ulsch_harq->ulsch_pdu.start_symbol_index;
          uint8_t symbol_end = symbol_start + ulsch_harq->ulsch_pdu.nr_of_symbols;
          VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_NR_RX_PUSCH,1);
	  start_meas(&gNB->rx_pusch_stats);
	  for(uint8_t symbol = symbol_start; symbol < symbol_end; symbol++) {
	    no_sig = nr_rx_pusch(gNB, ULSCH_id, frame_rx, slot_rx, symbol, harq_pid);
            if (no_sig) {
              LOG_I(PHY, "PUSCH not detected in symbol %d\n",symbol);
              nr_fill_indication(gNB,frame_rx, slot_rx, ULSCH_id, harq_pid, 1);
              return;
            }
	  }
	  stop_meas(&gNB->rx_pusch_stats);
          VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_NR_RX_PUSCH,0);
          //LOG_M("rxdataF_comp.m","rxF_comp",gNB->pusch_vars[0]->rxdataF_comp[0],6900,1,1);
          //LOG_M("rxdataF_ext.m","rxF_ext",gNB->pusch_vars[0]->rxdataF_ext[0],6900,1,1);
          VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_NR_ULSCH_PROCEDURES_RX,1);
          nr_ulsch_procedures(gNB, frame_rx, slot_rx, ULSCH_id, harq_pid);
          VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_NR_ULSCH_PROCEDURES_RX,0);
          break;
        }
      }
    }
  }
  // figure out a better way to choose slot_rx, 19 is ok for a particular TDD configuration with 30kHz SCS
  if ((frame_rx&127) == 0 && slot_rx==19) dump_pusch_stats(gNB);

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_gNB_UESPEC_RX,0);
}
