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

/** yunsdr_lib.h
 *
 * Author: eric
 * base on bladerf_lib.h
 */

#include "yunsdr_api_ss.h"
#include "common_lib.h"

/** @addtogroup _YUNSDR_PHY_RF_INTERFACE_
 * @{
 */

/*! \brief YunSDR specific data structure */ 
typedef struct {

  //! opaque YunSDR device struct. An empty ("") or NULL device identifier will result in the first encountered device being opened (using the first discovered backend)
  YUNSDR_DESCRIPTOR *dev;
  int16_t *rx_buffer;
  int16_t *tx_buffer;
  //! Sample rate
  unsigned int sample_rate;

  int rx_num_channels;
  int tx_num_channels;

  // --------------------------------
  // Debug and output control
  // --------------------------------
  //! Number of underflows
  int num_underflows;
  //! Number of overflows
  int num_overflows;
  //! number of RX errors
  int num_rx_errors;
  //! Number of TX errors
  int num_tx_errors;

  //! timestamp of current TX
  uint64_t tx_current_ts;
  //! timestamp of current RX
  uint64_t rx_current_ts;
  //! number of TX samples
  uint64_t tx_nsamps;
  //! number of RX samples
  uint64_t rx_nsamps;
  //! number of TX count
  uint64_t tx_count;
  //! number of RX count
  uint64_t rx_count;
  //! timestamp of RX packet
  openair0_timestamp rx_timestamp;

} yunsdr_state_t;

/*! \brief get current timestamp
 *\param device the hardware to use 
 */
openair0_timestamp trx_get_timestamp(openair0_device *device);

/*@}*/
