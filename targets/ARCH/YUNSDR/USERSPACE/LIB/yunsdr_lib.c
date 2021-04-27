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
#include <math.h>
#include "yunsdr_lib.h"
#include "rf_helper.h"
#include "common/utils/LOG/log.h"

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

#define RX_MTU         30720
#define BUFFER_SIZE    (122880 * 10 * sizeof(int))
#define NCHAN_PER_DEV  4
static void *cache_buf[NCHAN_PER_DEV];
static void *iq_buf[NCHAN_PER_DEV];
static uint32_t remain = 0;

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

    LOG_I(HW, "[yunsdr] Start yunsdr ...\n");
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
static int trx_yunsdr_write(openair0_device *device,openair0_timestamp timestamp, void **buff, int nsamps, int cc, int flags) {

    int status;
    yunsdr_state_t *yunsdr = (yunsdr_state_t*)device->priv;

#ifdef __GNUC__
    __sync_fetch_and_add(&transmiting, 1);
#endif
#ifdef __AVX2__
    __m256i a, *b;
    int len = nsamps * 2;
    int16_t *iq = buff[0];

    while (len >= 16) {
        a = *(__m256i *)&iq[0];
        b = (__m256i *)&iq[0];
        *b = _mm256_slli_epi16(a, 4);
        iq += 16;
        len -= 16;
    }
#else
    __m128i a, *b;
    int len = nsamps * 2;
    int16_t *iq = buff[0];

    while (len >= 8) {
        a = *(__m128i *)&iq[0];
        b = (__m128i *)&iq[0];
        *b = _mm_slli_epi16(a, 4);
        iq += 8;
        len -= 8;
    }
#endif
    /* remaining data */
    while (len != 0) {
        iq[0] <<= 4;
        iq++;
        len--;
    }

    status = yunsdr_write_samples_multiport(yunsdr->dev, (const void **)buff, nsamps, channel_to_mask(yunsdr->tx_num_channels), timestamp, 0);
    if (status < 0) {
        yunsdr->num_tx_errors++;
        LOG_E(HW, "[yunsdr] Failed to TX samples\n");
        exit(-1);
    }
#ifdef __GNUC__
    __sync_fetch_and_sub(&transmiting, 1);
#endif
    //LOG_D(HW, "Provided TX timestamp: %u, nsamps: %u\n", ptimestamp, nsamps);

    yunsdr->tx_current_ts = timestamp;
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

    if(remain == 0) {
        int recv = 0;
        if(nsamps % RX_MTU)
            recv = (nsamps / RX_MTU + 1) * RX_MTU;
        else
            recv = nsamps;
        timestamp = 0L;
        status = yunsdr_read_samples_multiport(yunsdr->dev, iq_buf, recv, channel_to_mask(yunsdr->rx_num_channels), &timestamp);
        if (status < 0) {
            LOG_E(HW, "[yunsdr] Failed to read samples %d\n", nsamps);
            yunsdr->num_rx_errors++;
            exit(-1);
        }
        for(int i = 0; i < yunsdr->rx_num_channels; i++)
            memcpy(buff[i], iq_buf[i], nsamps * 4);
        if(recv > nsamps) {
            for(int i = 0; i < yunsdr->rx_num_channels; i++)
                memcpy(cache_buf[i], iq_buf[i] + nsamps * 4, (recv - nsamps) * 4);
            remain = recv - nsamps;
        }
        *(uint64_t *)ptimestamp = timestamp;
        yunsdr->rx_current_ts = timestamp + nsamps;
        //LOG_D(HW, "case 0: Current RX timestamp  %"PRIu64", hw ts %"PRIu64", nsamps %u, remain %u, recv: %u\n",  *ptimestamp, timestamp, nsamps, remain, recv);
    } else if(remain >= nsamps) {
        for(int i = 0; i < yunsdr->rx_num_channels; i++)
            memcpy(buff[i], cache_buf[i], nsamps * 4);
        remain -= nsamps;
        if(remain > 0) {
            for(int i = 0; i < yunsdr->rx_num_channels; i++)
                memmove(cache_buf[i], cache_buf[i] + nsamps * 4, remain * 4);
        }
        *(uint64_t *)ptimestamp = yunsdr->rx_current_ts;
        yunsdr->rx_current_ts += nsamps;
        //LOG_D(HW, "case 1: Current RX timestamp  %"PRIu64", nsamps %u, remain %u\n",  *ptimestamp, nsamps, remain);
    } else {
        int recv;
        if(remain + RX_MTU >= nsamps)
            recv = RX_MTU;
        else
            recv = (nsamps / RX_MTU + 1) * RX_MTU;
        timestamp = 0L;
        status = yunsdr_read_samples_multiport(yunsdr->dev, iq_buf, recv, channel_to_mask(yunsdr->rx_num_channels), &timestamp);
        if (status < 0) {
            LOG_E(HW, "[yunsdr] Failed to read samples %d\n", nsamps);
            yunsdr->num_rx_errors++;
            exit(-1);
        }
        if(timestamp != (yunsdr->rx_current_ts + remain)) {
            int overflow = timestamp - (yunsdr->rx_current_ts + remain);
            //LOG_W(HW, "Rx overflow %u samples\n", overflow);
            remain += overflow;
        }
        for(int i = 0; i < yunsdr->rx_num_channels; i++)
            memcpy(cache_buf[i] + remain * 4, iq_buf[i], recv * 4);
        for(int i = 0; i < yunsdr->rx_num_channels; i++)
            memcpy(buff[i], cache_buf[i], nsamps * 4);
        remain = recv + remain - nsamps;
        for(int i = 0; i < yunsdr->rx_num_channels; i++)
            memmove(cache_buf[i], cache_buf[i] + nsamps * 4, remain * 4);

        *(uint64_t *)ptimestamp = yunsdr->rx_current_ts;
        yunsdr->rx_current_ts += nsamps;
        //LOG_D(HW, "case 2: Current RX timestamp  %"PRIu64", hw ts %"PRIu64", nsamps %u, remain %u, recv: %u\n",  *ptimestamp, timestamp, nsamps, remain, recv);
    }

#ifdef __GNUC__
    __sync_fetch_and_sub(&recving, 1);
#endif
    //LOG_D(HW, "Current RX timestamp  %"PRIu64", nsamps %u\n",  *ptimestamp, nsamps);
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
    LOG_I(HW, ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    uint32_t count = 0;
    yunsdr_get_channel_event(yunsdr->dev, TX_CHANNEL_TIMEOUT, 1, &count);
    LOG_I(HW, "[yunsdr] TX%d Channel timeout: %u\n", 1, count);
    yunsdr_get_channel_event(yunsdr->dev, RX_CHANNEL_OVERFLOW, 1, &count);
    LOG_I(HW, "[yunsdr] RX%d Channel overflow: %u\n", 1, count);
    LOG_I(HW, "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n");

    yunsdr_close_device(yunsdr->dev);
    //exit(1);
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
        LOG_E(HW, "[yunsdr] Failed to set TX frequency\n");
    } else
        LOG_I(HW, "[yunsdr] set TX frequency to %lu\n",(uint64_t)(openair0_cfg->tx_freq[0]));

    if ((status = yunsdr_set_rx_lo_freq(yunsdr->dev, 0, (uint64_t)(openair0_cfg->rx_freq[0]))) < 0) {
        LOG_E(HW, "[yunsdr] Failed to set RX frequency\n");
    } else
        LOG_I(HW, "[yunsdr] set RX frequency to %lu\n",(uint64_t)(openair0_cfg->rx_freq[0]));

    return(0);

}

/*! \brief Set Gains (TX/RX)
 * \param device the hardware to use
 * \param openair0_cfg openair0 Config structure
 * \returns 0 in success
 */
int trx_yunsdr_set_gains(openair0_device* device, openair0_config_t *openair0_cfg) {

    int ret = 0;
    yunsdr_state_t *yunsdr = (yunsdr_state_t *)device->priv;

    if (openair0_cfg->rx_gain[0] > 65+openair0_cfg->rx_gain_offset[0]) {
        LOG_E(HW, "[yunsdr] Reduce RX Gain 0 by %f dB\n", openair0_cfg->rx_gain[0] - openair0_cfg->rx_gain_offset[0] - 65);
        return -1;
    }

    if ((ret = yunsdr_set_rx1_rf_gain(yunsdr->dev, 0, (uint32_t)(openair0_cfg->rx_gain[0] > 65?65:openair0_cfg->rx_gain[0]))) < 0) {
        LOG_I(HW, "[yunsdr] Failed to set RX1 gain\n");
    } else
        LOG_I(HW, "[yunsdr] set RX1 gain to %u\n",(uint32_t)(openair0_cfg->rx_gain[0]));

    if(yunsdr->rx_num_channels > 1) {
        if ((ret = yunsdr_set_rx2_rf_gain(yunsdr->dev, 0, (uint32_t)(openair0_cfg->rx_gain[1] > 65?65:openair0_cfg->rx_gain[1]))) < 0) {
            LOG_E(HW, "[yunsdr] Failed to set RX2 gain\n");
        } else
            LOG_I(HW, "[yunsdr] set RX gain to %u\n",(uint32_t)(openair0_cfg->rx_gain[1]));
    }

    int tx_gain = ((uint32_t)openair0_cfg->tx_gain[0] > 90?90:(uint32_t)openair0_cfg->tx_gain[0]);
    if ((ret = yunsdr_set_tx1_attenuation(yunsdr->dev, 0, (90 - tx_gain) * 1000)) < 0) {
        LOG_E(HW, "[yunsdr] Failed to set TX1 gain\n");
    } else
        LOG_I(HW, "[yunsdr] set the TX1 gain to %d\n", (uint32_t)openair0_cfg->tx_gain[0]);

    if(yunsdr->tx_num_channels > 1) {
        tx_gain = ((uint32_t)openair0_cfg->tx_gain[1] > 90?90:(uint32_t)openair0_cfg->tx_gain[1]);
        if ((ret = yunsdr_set_tx2_attenuation(yunsdr->dev, 0, (90 - tx_gain) * 1000)) < 0) {
            LOG_E(HW, "[yunsdr] Failed to set TX2 gain\n");
        } else
            LOG_I(HW, "[yunsdr] set the TX2 gain to %d\n", (uint32_t)openair0_cfg->tx_gain[1]);
    }

    return(ret);
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

    LOG_I(HW, "[yunsdr] openair0_cfg[0].sdr_addrs == '%s'\n", openair0_cfg[0].sdr_addrs);
    LOG_I(HW, "[yunsdr] openair0_cfg[0].rx_num_channels == '%d'\n", openair0_cfg[0].rx_num_channels);
    LOG_I(HW, "[yunsdr] openair0_cfg[0].tx_num_channels == '%d'\n", openair0_cfg[0].tx_num_channels);

    // init required params
    switch ((int)openair0_cfg->sample_rate) {
    case 122880000:
        openair0_cfg->samples_per_packet    = 122880;
        openair0_cfg->tx_sample_advance     = 70;
        openair0_cfg[0].tx_bw               = 100e6;
        openair0_cfg[0].rx_bw               = 100e6;
        break;
    case 61440000:
        openair0_cfg->samples_per_packet    = 61440;
        openair0_cfg->tx_sample_advance     = 70;
        openair0_cfg[0].tx_bw               = 40e6;
        openair0_cfg[0].rx_bw               = 40e6;
        break;
    case 30720000:
        openair0_cfg->samples_per_packet    = 30720;
        openair0_cfg->tx_sample_advance     = 70;
        openair0_cfg[0].tx_bw               = 20e6;
        openair0_cfg[0].rx_bw               = 20e6;
        break;
    case 15360000:
        openair0_cfg->samples_per_packet    = 15360;
        openair0_cfg->tx_sample_advance     = 68;
        openair0_cfg[0].tx_bw               = 10e6;
        openair0_cfg[0].rx_bw               = 10e6;
        break;
    case 7680000:
        openair0_cfg->samples_per_packet    = 7680;
        openair0_cfg->tx_sample_advance     = 34;
        openair0_cfg[0].tx_bw               = 5e6;
        openair0_cfg[0].rx_bw               = 5e6;
        break;
    case 1920000:
        openair0_cfg->samples_per_packet    = 1920;
        openair0_cfg->tx_sample_advance     = 9;
        openair0_cfg[0].tx_bw               = 1.25e6;
        openair0_cfg[0].rx_bw               = 1.25e6;
        break;
    default:
        LOG_I(HW, "[yunsdr] Error: unknown sampling rate %f\n", openair0_cfg->sample_rate);
        free(yunsdr);
        exit(-1);
        break;
    }
    //openair0_cfg->iq_txshift = 2;
    //openair0_cfg->iq_rxrescale = 14; /*not sure*/ //FIXME: adjust to yunsdr
    yunsdr->sample_rate = (unsigned int)openair0_cfg->sample_rate;
    LOG_I(HW, "[yunsdr] sampling_rate %d\n", yunsdr->sample_rate);
    yunsdr->rx_num_channels = openair0_cfg[0].rx_num_channels;
    yunsdr->tx_num_channels = openair0_cfg[0].tx_num_channels;

    int auxdac1 = 0;
    char args[64];
    if (openair0_cfg[0].sdr_addrs == NULL) {
        strcpy(args, "dev=pcie:0");
    } else {
        strcpy(args, openair0_cfg[0].sdr_addrs);
    }

    char dev_str[64];
    const char dev_arg[] = "dev=";
    char *dev_ptr = strstr(args, dev_arg);
    if(dev_ptr) {
        copy_subdev_string(dev_str, dev_ptr + strlen(dev_arg));
        remove_substring(args, dev_arg);
        remove_substring(args, dev_str);
        LOG_I(HW, "[yunsdr] Using %s\n", dev_str);
    }

    const char auxdac1_arg[] = "auxdac1=";
    char auxdac1_str[64] = {0};
    char *auxdac1_ptr = strstr(args, auxdac1_arg);
    if(auxdac1_ptr) {
        copy_subdev_string(auxdac1_str, auxdac1_ptr + strlen(auxdac1_arg));
        remove_substring(args, auxdac1_arg);
        remove_substring(args, auxdac1_str);
        auxdac1 = atoi(auxdac1_str);
        LOG_I(HW, "[yunsdr] Setting auxdac1:%u\n", auxdac1);
    }

    if ((yunsdr->dev = yunsdr_open_device(dev_str)) == NULL ) {
        LOG_E(HW, "[yunsdr] Failed to open yunsdr\n");
        free(yunsdr);
        return -1;
    }

    LOG_I(HW, "[yunsdr] Initializing openair0_device\n");
    switch (openair0_cfg[0].clock_source) {
    case external:
        LOG_I(HW, "[yunsdr] clock_source: external\n");
        yunsdr_set_ref_clock (yunsdr->dev, 0, EXTERNAL_REFERENCE);
        yunsdr_set_pps_select (yunsdr->dev, 0, PPS_EXTERNAL_EN);
        break;
    case gpsdo:
        LOG_I(HW, "[yunsdr] clock_source: gpsdo\n");
        break;
    case internal:
    default:
        yunsdr_set_ref_clock (yunsdr->dev, 0, INTERNAL_REFERENCE);
        yunsdr_set_pps_select (yunsdr->dev, 0, PPS_INTERNAL_EN);
        //yunsdr_set_vco_select (yunsdr->dev, 0, AUXDAC1);
        LOG_I(HW, "[yunsdr] clock_source: internal\n");
        break;
    }
    yunsdr_set_auxdac1 (yunsdr->dev, 0, auxdac1);
    yunsdr_set_duplex_select (yunsdr->dev, 0, FDD);
    yunsdr_set_trxsw_fpga_enable(yunsdr->dev, 0, 0);
    yunsdr_set_rx_ant_enable (yunsdr->dev, 0, 1);
    yunsdr_set_tx_fir_en_dis (yunsdr->dev, 0, 0);
    yunsdr_set_rx_fir_en_dis (yunsdr->dev, 0, 0);

    // RX port Initialize
    if ((status = yunsdr_set_rx_lo_freq(yunsdr->dev, 0, (uint64_t)(openair0_cfg->rx_freq[0]))) < 0) {
        LOG_E(HW, "[yunsdr] Failed to set RX frequency\n");
    } else
        LOG_I(HW, "[yunsdr] set RX frequency to %lu\n",(uint64_t)(openair0_cfg->rx_freq[0]));
    if ((status = yunsdr_set_rx_sampling_freq(yunsdr->dev, 0, (uint32_t)(openair0_cfg->sample_rate))) < 0) {
        LOG_E(HW, "[yunsdr] Failed to set RX sample rate\n");
    } else
        LOG_I(HW, "[yunsdr] set RX sample rate to %u\n", (uint32_t)(openair0_cfg->sample_rate));
    if ((status = yunsdr_set_rx_rf_bandwidth(yunsdr->dev, 0, (uint32_t)(openair0_cfg->rx_bw))) < 0) {
        LOG_E(HW, "[yunsdr] Failed to set RX bandwidth\n");
    } else
        LOG_I(HW, "[yunsdr] set RX bandwidth to %u\n",(uint32_t)(openair0_cfg->rx_bw));

    if ((status = yunsdr_set_rx1_gain_control_mode(yunsdr->dev, 0, 0)) < 0){
        LOG_E(HW, "[yunsdr] Failed to set RX1 Gain Control Mode\n");
    } else
        LOG_I(HW, "[yunsdr] set RX1 Gain Control Mode MGC\n");

    if ((status = yunsdr_set_rx1_rf_gain(yunsdr->dev, 0, (uint32_t)(openair0_cfg->rx_gain[0] > 65?65:openair0_cfg->rx_gain[0]))) < 0) {
        LOG_I(HW, "[yunsdr] Failed to set RX1 gain\n");
    } else
        LOG_I(HW, "[yunsdr] set RX1 gain to %u\n",(uint32_t)(openair0_cfg->rx_gain[0]));

    if(yunsdr->rx_num_channels > 1) {
        if ((status = yunsdr_set_rx2_gain_control_mode(yunsdr->dev, 0, 0)) < 0){
            LOG_E(HW, "[yunsdr] Failed to set RX2 Gain Control Mode\n");
        } else
            LOG_I(HW, "[yunsdr] set RX2 Gain Control Mode MGC\n");

        if ((status = yunsdr_set_rx2_rf_gain(yunsdr->dev, 0, (uint32_t)(openair0_cfg->rx_gain[1] > 65?65:openair0_cfg->rx_gain[1]))) < 0) {
            LOG_E(HW, "[yunsdr] Failed to set RX2 gain\n");
        } else
            LOG_I(HW, "[yunsdr] set RX2 gain to %u\n",(uint32_t)(openair0_cfg->rx_gain[1]));
    }

    // TX port Initialize
    if ((status = yunsdr_set_tx_lo_freq(yunsdr->dev, 0, (uint64_t)openair0_cfg->tx_freq[0])) < 0) {
        LOG_E(HW, "[yunsdr] Failed to set TX frequency\n");
    } else
        LOG_I(HW, "[yunsdr] set TX Frequency to %lu\n", (uint64_t)openair0_cfg->tx_freq[0]);

    if ((status = yunsdr_set_tx_sampling_freq(yunsdr->dev, 0, (uint32_t)openair0_cfg->sample_rate)) < 0) {
        LOG_E(HW, "[yunsdr] Failed to set TX sample rate\n");
    } else
        LOG_I(HW, "[yunsdr] set TX sampling rate to %u\n", (uint32_t)openair0_cfg->sample_rate);

    if ((status = yunsdr_set_tx_rf_bandwidth(yunsdr->dev, 0, (uint32_t)openair0_cfg->tx_bw)) <0) {
        LOG_E(HW, "[yunsdr] Failed to set TX bandwidth\n");
    } else
        LOG_I(HW, "[yunsdr] set TX bandwidth to %u\n", (uint32_t)openair0_cfg->tx_bw);

    int tx_gain = ((uint32_t)openair0_cfg->tx_gain[0] > 90?90:(uint32_t)openair0_cfg->tx_gain[0]);
    if ((status = yunsdr_set_tx1_attenuation(yunsdr->dev, 0, (90 - tx_gain) * 1000)) < 0) {
        LOG_E(HW, "[yunsdr] Failed to set TX1 gain\n");
    } else
        LOG_I(HW, "[yunsdr] set the TX1 gain to %d\n", (uint32_t)openair0_cfg->tx_gain[0]);

    if(yunsdr->tx_num_channels > 1) {
        tx_gain = ((uint32_t)openair0_cfg->tx_gain[1] > 90?90:(uint32_t)openair0_cfg->tx_gain[1]);
        if ((status = yunsdr_set_tx2_attenuation(yunsdr->dev, 0, (90 - tx_gain) * 1000)) < 0) {
            LOG_E(HW, "[yunsdr] Failed to set TX2 gain\n");
        } else
            LOG_I(HW, "[yunsdr] set the TX2 gain to %d\n", (uint32_t)openair0_cfg->tx_gain[1]);
    }

    yunsdr_enable_timestamp(yunsdr->dev, 0, 0);
    usleep(5);
    yunsdr_enable_timestamp(yunsdr->dev, 0, 1);

    for(int i = 0; i < NCHAN_PER_DEV; i++) {
        int ret = posix_memalign((void **)&cache_buf[i], 4096, BUFFER_SIZE);
        if(ret) {
            LOG_I(HW, "Failed to alloc memory\n");
            return -1;
        }
        ret = posix_memalign((void **)&iq_buf[i], 4096, BUFFER_SIZE);
        if(ret) {
            LOG_I(HW, "Failed to alloc memory\n");
            return -1;
        }
    }

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
