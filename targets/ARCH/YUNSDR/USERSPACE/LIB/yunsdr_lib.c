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

/** yunsdr_lib.c
 *
 * Author: eric
 * base on bladerf_lib.c
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <inttypes.h>
#include "yunsdr_lib.h"
#include "math.h"

/** @addtogroup _YUNSDR_PHY_RF_INTERFACE_
 * @{
 */

#ifdef __SSE4_1__
#  include <smmintrin.h>
#endif

#ifdef __AVX2__
#  include <immintrin.h>
#endif

//! Number of YUNSDR devices
int num_devices = 0;
#ifdef __GNUC__
static int recving = 0;
static int transmiting = 0;
#endif
static bool running = false;

static inline int channel_to_mask(int channel_count)
{
    uint8_t ch_mask;
    switch (channel_count) {
    case 4:
        ch_mask = 0xf;break;
    case 3:
        ch_mask = 0x7;break;
    case 2:
        ch_mask = 0x3;break;
    case 1:
        ch_mask = 0x1;break;
    default:
        ch_mask = 0x1;break;
    }

    return ch_mask;
}

/*! \brief get current timestamp
 *\param device the hardware to use
 *\returns timestamp of YunSDR
 */

openair0_timestamp trx_get_timestamp(openair0_device *device) {
    return 0;
}

/*! \brief Start yunsdr
 * \param device the hardware to use
 * \returns 0 on success
 */
int trx_yunsdr_start(openair0_device *device) {

    printf("[yunsdr] Start yunsdr ...\n");
    running = true;

    return 0;
}

/*! \brief Called to send samples to the yunsdr RF target
  \param device pointer to the device structure specific to the RF hardware target
  \param timestamp The timestamp at whicch the first sample MUST be sent
  \param buff Buffer which holds the samples
  \param nsamps number of samples to be sent
  \param cc index of the component carrier
  \param flags Ignored for the moment
  \returns 0 on success
  */
static int trx_yunsdr_write(openair0_device *device,openair0_timestamp ptimestamp, void **buff, int nsamps, int cc, int flags) {

    int status;
    yunsdr_state_t *yunsdr = (yunsdr_state_t*)device->priv;
    uint64_t timestamp = 0L;

    if(flags <= 0)
        timestamp = 0;
    else
        timestamp = (uint64_t)ptimestamp;
#ifdef __GNUC__
    __sync_fetch_and_add(&transmiting, 1);
#endif
    status = yunsdr_write_samples_multiport(yunsdr->dev, (const void **)buff, nsamps, channel_to_mask(yunsdr->tx_num_channels), timestamp, 0);
    if (status < 0) {
        yunsdr->num_tx_errors++;
        printf("[yunsdr] Failed to TX samples\n");
        exit(-1);
    }
#ifdef __GNUC__
    __sync_fetch_and_sub(&transmiting, 1);
#endif
    //printf("Provided TX timestamp: %u, nsamps: %u\n", ptimestamp, nsamps);

    yunsdr->tx_current_ts = ptimestamp;
    yunsdr->tx_nsamps += nsamps;
    yunsdr->tx_count++;

    return nsamps;
}

/*! \brief Receive samples from hardware.
 * Read \ref nsamps samples from each channel to buffers. buff[0] is the array for
 * the first channel. *ptimestamp is the time at which the first sample
 * was received.
 * \param device the hardware to use
 * \param[out] ptimestamp the time at which the first sample was received.
 * \param[out] buff An array of pointers to buffers for received samples. The buffers must be large enough to hold the number of samples \ref nsamps.
 * \param nsamps Number of samples. One sample is 2 byte I + 2 byte Q => 4 byte.
 * \param cc  Index of component carrier
 * \returns number of samples read
 */
static int trx_yunsdr_read(openair0_device *device, openair0_timestamp *ptimestamp, void **buff, int nsamps, int cc) {

    int status;
    yunsdr_state_t *yunsdr = (yunsdr_state_t *)device->priv;
    uint64_t timestamp = 0L;

#ifdef __GNUC__
    __sync_fetch_and_add(&recving, 1);
#endif
    timestamp = 0L;
    status = yunsdr_read_samples_multiport(yunsdr->dev, buff, nsamps, channel_to_mask(yunsdr->rx_num_channels), &timestamp);
    if (status < 0) {
        printf("[yunsdr] Failed to read samples %d\n", nsamps);
        yunsdr->num_rx_errors++;
        exit(-1);
    }
#ifdef __GNUC__
    __sync_fetch_and_sub(&recving, 1);
#endif
    //printf("Current RX timestamp  %"PRIu64", nsamps %u\n",  *ptimestamp, nsamps);
    *(uint64_t *)ptimestamp = timestamp;
    yunsdr->rx_current_ts = *ptimestamp;
    yunsdr->rx_nsamps += nsamps;
    yunsdr->rx_count++;

    return nsamps;

}

/*! \brief Terminate operation of the yunsdr transceiver -- free all associated resources
 * \param device the hardware to use
 */
void trx_yunsdr_end(openair0_device *device) {

    yunsdr_state_t *yunsdr = (yunsdr_state_t*)device->priv;

    if(!running)
        return;
    running = false;

#ifdef __GNUC__
    while(__sync_and_and_fetch(&recving, 1) ||
            __sync_and_and_fetch(&transmiting, 1))
        usleep(50000);
#endif
    printf(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    uint32_t count = 0;
    yunsdr_get_channel_event(yunsdr->dev, TX_CHANNEL_TIMEOUT, 1, &count);
    printf("TX%d Channel timeout: %u\n", 1, count);
    yunsdr_get_channel_event(yunsdr->dev, RX_CHANNEL_OVERFLOW, 1, &count);
    printf("RX%d Channel overflow: %u\n", 1, count);
    printf("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");

    yunsdr_close_device(yunsdr->dev);
    exit(1);
}

/*! \brief print the yunsdr statistics 
 * \param device the hardware to use
 * \returns  0 on success
 */
int trx_yunsdr_get_stats(openair0_device* device) {

    return(0);

}

/*! \brief Reset the yunsdr statistics 
 * \param device the hardware to use
 * \returns  0 on success
 */
int trx_yunsdr_reset_stats(openair0_device* device) {

    return(0);

}

/*! \brief Stop yunsdr
 * \param card the hardware to use
 * \returns 0 in success 
 */
int trx_yunsdr_stop(openair0_device* device) {

    return(0);

}

/*! \brief Set frequencies (TX/RX)
 * \param device the hardware to use
 * \param openair0_cfg1 openair0 Config structure (ignored. It is there to comply with RF common API)
 * \param exmimo_dump_config (ignored)
 * \returns 0 in success 
 */
int trx_yunsdr_set_freq(openair0_device* device, openair0_config_t *openair0_cfg1,int exmimo_dump_config) {

    int status;
    yunsdr_state_t *yunsdr = (yunsdr_state_t *)device->priv;
    openair0_config_t *openair0_cfg = (openair0_config_t *)device->openair0_cfg;

    if ((status = yunsdr_set_tx_lo_freq(yunsdr->dev, 0, (uint64_t)(openair0_cfg->tx_freq[0]))) < 0) {
        fprintf(stderr,"[yunsdr] Failed to set TX frequency\n");
    } else
        printf("[yunsdr] set TX frequency to %lu\n",(uint64_t)(openair0_cfg->tx_freq[0]));

    if ((status = yunsdr_set_rx_lo_freq(yunsdr->dev, 0, (uint64_t)(openair0_cfg->rx_freq[0]))) < 0) {
        fprintf(stderr,"[yunsdr] Failed to set RX frequency\n");
    } else
        printf("[yunsdr] set RX frequency to %lu\n",(uint64_t)(openair0_cfg->rx_freq[0]));

    return(0);

}

/*! \brief Set Gains (TX/RX)
 * \param device the hardware to use
 * \param openair0_cfg openair0 Config structure
 * \returns 0 in success
 */
int trx_yunsdr_set_gains(openair0_device* device, openair0_config_t *openair0_cfg) {

    return(0);

}

/*! \brief calibration table for yunsdr */
rx_gain_calib_table_t calib_table_fx4[] = {
    {2300000000.0,53.5},
    {1880000000.0,57.0},
    {816000000.0,73.0},
    {-1,0}};

/*! \brief set RX gain offset from calibration table
 * \param openair0_cfg RF frontend parameters set by application
 * \param chain_index RF chain ID
 */
void set_rx_gain_offset(openair0_config_t *openair0_cfg, int chain_index) {

    int i = 0;
    // loop through calibration table to find best adjustment factor for RX frequency
    double min_diff = 6e9,diff;

    while (openair0_cfg->rx_gain_calib_table[i].freq > 0) {
        diff = fabs(openair0_cfg->rx_freq[chain_index] - openair0_cfg->rx_gain_calib_table[i].freq);
        printf("cal %d: freq %f, offset %f, diff %f\n",
                i,
                openair0_cfg->rx_gain_calib_table[i].freq,
                openair0_cfg->rx_gain_calib_table[i].offset, diff);
        if (min_diff > diff) {
            min_diff = diff;
            openair0_cfg->rx_gain_offset[chain_index] = openair0_cfg->rx_gain_calib_table[i].offset;
        }
        i++;
    }

}

/*! \brief Initialize Openair yunsdr target. It returns 0 if OK
 * \param device the hardware to use
 * \param openair0_cfg RF frontend parameters set by application
 * \returns 0 on success
 */
int device_init(openair0_device *device, openair0_config_t *openair0_cfg) {

    int status;

    yunsdr_state_t *yunsdr = (yunsdr_state_t*)malloc(sizeof(yunsdr_state_t));
    memset(yunsdr, 0, sizeof(yunsdr_state_t));

    printf("[yunsdr] openair0_cfg[0].sdr_addrs == '%s'\n", openair0_cfg[0].sdr_addrs);
    printf("[yunsdr] openair0_cfg[0].rx_num_channels == '%d'\n", openair0_cfg[0].rx_num_channels);
    printf("[yunsdr] openair0_cfg[0].tx_num_channels == '%d'\n", openair0_cfg[0].tx_num_channels);

    // init required params
    switch ((int)openair0_cfg->sample_rate) {
    case 122880000:
        openair0_cfg->samples_per_packet    = 122880;
        openair0_cfg->tx_sample_advance     = 132;
        break;
    case 61440000:
        openair0_cfg->samples_per_packet    = 61440;
        openair0_cfg->tx_sample_advance     = 132;
        break;
    case 30720000:
        openair0_cfg->samples_per_packet    = 30720;
        openair0_cfg->tx_sample_advance     = 132;
        break;
    case 15360000:
        openair0_cfg->samples_per_packet    = 15360;
        openair0_cfg->tx_sample_advance     = 68;
        break;
    case 7680000:
        openair0_cfg->samples_per_packet    = 7680;
        openair0_cfg->tx_sample_advance     = 34;
        break;
    case 1920000:
        openair0_cfg->samples_per_packet    = 1920;
        openair0_cfg->tx_sample_advance     = 9;
        break;
    default:
        printf("[yunsdr] Error: unknown sampling rate %f\n", openair0_cfg->sample_rate);
        free(yunsdr);
        exit(-1);
        break;
    }
    openair0_cfg->iq_txshift = 0;
    openair0_cfg->iq_rxrescale = 14; /*not sure*/ //FIXME: adjust to yunsdr
    yunsdr->sample_rate = (unsigned int)openair0_cfg->sample_rate;
    printf("[yunsdr] sampling_rate %d\n", yunsdr->sample_rate);
    yunsdr->rx_num_channels = openair0_cfg[0].rx_num_channels;
    yunsdr->tx_num_channels = openair0_cfg[0].tx_num_channels;

    char args[64];
    if (openair0_cfg[0].sdr_addrs == NULL) {
        strcpy(args, "pcie:0");
    } else {
        strcpy(args, openair0_cfg[0].sdr_addrs);
    }

    if ((yunsdr->dev = yunsdr_open_device(args)) == NULL ) {
        fprintf(stderr,"[yunsdr] Failed to open yunsdr\n");
        free(yunsdr);
        return -1;
    }

    printf("[yunsdr] Initializing openair0_device\n");
    switch (openair0_cfg[0].clock_source) {
    case external:
        printf("[yunsdr] clock_source: external\n");
        yunsdr_set_ref_clock (yunsdr->dev, 0, EXTERNAL_REFERENCE);
        yunsdr_set_pps_select (yunsdr->dev, 0, PPS_EXTERNAL_EN);
        break;
    case gpsdo:
        printf("[yunsdr] clock_source: gpsdo\n");
        break;
    case internal:
    default:
        yunsdr_set_ref_clock (yunsdr->dev, 0, INTERNAL_REFERENCE);
        yunsdr_set_pps_select (yunsdr->dev, 0, PPS_INTERNAL_EN);
        yunsdr_set_vco_select (yunsdr->dev, 0, AUXDAC1);
        //yunsdr_set_auxdac1 (yunsdr->dev, 0, 100);
        printf("[yunsdr] clock_source: internal\n");
        break;
    }
    yunsdr_set_auxdac1 (yunsdr->dev, 0, 1470);
    yunsdr_set_duplex_select (yunsdr->dev, 0, FDD);
    yunsdr_set_trxsw_fpga_enable(yunsdr->dev, 0, 0);
    yunsdr_set_rx_ant_enable (yunsdr->dev, 0, 1);
    yunsdr_set_tx_fir_en_dis (yunsdr->dev, 0, 0);
    yunsdr_set_rx_fir_en_dis (yunsdr->dev, 0, 0);

    // RX port Initialize
    if ((status = yunsdr_set_rx_lo_freq(yunsdr->dev, 0, (uint64_t)(openair0_cfg->rx_freq[0]))) < 0) {
        fprintf(stderr,"[yunsdr] Failed to set RX frequency\n");
    } else
        printf("[yunsdr] set RX frequency to %lu\n",(uint64_t)(openair0_cfg->rx_freq[0]));
    if ((status = yunsdr_set_rx_sampling_freq(yunsdr->dev, 0, (uint32_t)(openair0_cfg->sample_rate))) < 0) {
        fprintf(stderr,"[yunsdr] Failed to set RX sample rate\n");
    } else
        printf("[yunsdr] set RX sample rate to %u\n", (uint32_t)(openair0_cfg->sample_rate));
    if ((status = yunsdr_set_rx_rf_bandwidth(yunsdr->dev, 0, (uint32_t)(openair0_cfg->rx_bw*2))) < 0) {
        fprintf(stderr,"[yunsdr] Failed to set RX bandwidth\n");
    } else
        printf("[yunsdr] set RX bandwidth to %u\n",(uint32_t)(openair0_cfg->rx_bw*2));
    if ((status = yunsdr_set_rx1_gain_control_mode(yunsdr->dev, 0, 0)) < 0){
        fprintf(stderr,"[yunsdr] Failed to set RX Gain Control Mode\n");
    } else
        printf("[yunsdr] set RX Gain Control Mode MGC\n");

    //if ((status = yunsdr_set_rx1_rf_gain(yunsdr->dev, 0, (uint32_t)(openair0_cfg->rx_gain[0] > 65?65:openair0_cfg->rx_gain[0]))) < 0) {
    if ((status = yunsdr_set_rx1_rf_gain(yunsdr->dev, 0, 65)) < 0) {
        fprintf(stderr,"[yunsdr] Failed to set RX gain\n");
    } else
        printf("[yunsdr] set RX gain to %u\n",(uint32_t)(openair0_cfg->rx_gain[0]));

    // TX port Initialize
    if ((status = yunsdr_set_tx_lo_freq(yunsdr->dev, 0, (uint64_t)openair0_cfg->tx_freq[0])) < 0) {
        fprintf(stderr,"[yunsdr] Failed to set TX frequency\n");
    } else
        printf("[yunsdr] set TX Frequency to %lu\n", (uint64_t)openair0_cfg->tx_freq[0]);

    if ((status = yunsdr_set_tx_sampling_freq(yunsdr->dev, 0, (uint32_t)openair0_cfg->sample_rate)) < 0) {
        fprintf(stderr,"[yunsdr] Failed to set TX sample rate\n");
    } else
        printf("[yunsdr] set TX sampling rate to %u\n", (uint32_t)openair0_cfg->sample_rate);

    if ((status = yunsdr_set_tx_rf_bandwidth(yunsdr->dev, 0, (uint32_t)openair0_cfg->tx_bw*2)) <0) {
        fprintf(stderr, "[yunsdr] Failed to set TX bandwidth\n");
    } else
        printf("[yunsdr] set TX bandwidth to %u\n", (uint32_t)openair0_cfg->tx_bw*2);

    //if ((status = yunsdr_set_tx1_attenuation(yunsdr->dev, 0, (90 - (uint32_t)openair0_cfg->tx_gain[0])*1000)) < 0) {
    if ((status = yunsdr_set_tx1_attenuation(yunsdr->dev, 0, 5000)) < 0) {
        fprintf(stderr,"[yunsdr] Failed to set TX gain\n");
    } else
        printf("[yunsdr] set the TX gain to %d\n", (uint32_t)openair0_cfg->tx_gain[0]);

    yunsdr_enable_timestamp(yunsdr->dev, 0, 0);
    usleep(5);
    yunsdr_enable_timestamp(yunsdr->dev, 0, 1);
   
    device->Mod_id               = num_devices++;
    device->type                 = YUNSDR_DEV;
    device->trx_start_func       = trx_yunsdr_start;
    device->trx_end_func         = trx_yunsdr_end;
    device->trx_read_func        = trx_yunsdr_read;
    device->trx_write_func       = trx_yunsdr_write;
    device->trx_get_stats_func   = trx_yunsdr_get_stats;
    device->trx_reset_stats_func = trx_yunsdr_reset_stats;
    device->trx_stop_func        = trx_yunsdr_stop;
    device->trx_set_freq_func    = trx_yunsdr_set_freq;
    device->trx_set_gains_func   = trx_yunsdr_set_gains;
    device->openair0_cfg         = openair0_cfg;
    device->priv                 = (void *)yunsdr;

    return 0;
}

/*@}*/
