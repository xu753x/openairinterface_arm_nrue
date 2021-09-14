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


#include <stdio.h>
#include "common_lib.h"
#include "ethernet_lib.h"
#include "shared_buffers.h"
// #include "xran_lib_wrap.hpp"


typedef struct {
  eth_state_t           e;
  shared_buffers        buffers;
  rru_config_msg_type_t last_msg;
  int                   capabilities_sent;
  void                  *oran_priv;
} oran_eth_state_t;



char *msg_type(int t)
{
  static char *s[12] = {
    "RAU_tick",
    "RRU_capabilities",
    "RRU_config",
    "RRU_config_ok",
    "RRU_start",
    "RRU_stop",
    "RRU_sync_ok",
    "RRU_frame_resynch",
    "RRU_MSG_max_num",
    "RRU_check_sync",
    "RRU_config_update",
    "RRU_config_update_ok",
  };

  if (t < 0 || t > 11) return "UNKNOWN";
  return s[t];
}

#if 0
void xran_fh_rx_callback(void *pCallbackTag, xran_status_t status){
    rte_pause();
}
void xran_fh_srs_callback(void *pCallbackTag, xran_status_t status){
    rte_pause();
}
void xran_fh_rx_prach_callback(void *pCallbackTag, xran_status_t status){
    rte_pause();
}
#endif

int trx_oran_start(openair0_device *device)
{
   
	return 0;
}


void trx_oran_end(openair0_device *device)
{

}


int trx_oran_stop(openair0_device *device)
{
  
	return 0;
}

int trx_oran_set_freq(openair0_device* device,
                         openair0_config_t *openair0_cfg,
                         int exmimo_dump_config)
{
  
	return 0;
}

int trx_oran_set_gains(openair0_device* device,
                          openair0_config_t *openair0_cfg)
{
  
	return 0;
}

int trx_oran_get_stats(openair0_device* device)
{
   
	return 0;
}


int trx_oran_reset_stats(openair0_device* device)
{
  
	return 0;
}

int ethernet_tune(openair0_device *device,
                  unsigned int option,
                  int value)
{
  
	return 0;
}

int trx_oran_write_raw(openair0_device *device,
                          openair0_timestamp timestamp,
                          void **buff, int nsamps, int cc, int flags)
{
  
	return 0;
}

int trx_oran_read_raw(openair0_device *device,
                         openair0_timestamp *timestamp,
                         void **buff, int nsamps, int cc)
{
  
	return 0;
}

int trx_oran_ctlsend(openair0_device *device, void *msg, ssize_t msg_len)
{
  
	return 0;
}


int trx_oran_ctlrecv(openair0_device *device, void *msg, ssize_t msg_len)
{
  
	return 0;
}




/*This function reads the IQ samples from OAI, symbol by symbol.
 It also handles the shared buffers.                          */

void oran_fh_if4p5_south_in(RU_t *ru,
                               int *frame,
                               int *subframe)
{
	
}


void oran_fh_if4p5_south_out(RU_t *ru,
                                int frame,
                                int subframe,
                                uint64_t timestamp)
{
  
	
}

__attribute__((__visibility__("default")))
extern "C"
{
  int transport_init(openair0_device *device,
                     openair0_config_t *openair0_cfg,
                     eth_params_t * eth_params )
  {
    
	  return 0;
  }
}


























