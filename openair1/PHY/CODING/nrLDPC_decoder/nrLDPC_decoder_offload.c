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

/*!\file nrLDPC_decoder_offload.c
 * \brief Defines the LDPC decoder
 * \author openairinterface 
 * \date 12-06-2021
 * \version 1.0
 * \note
 * \warning
 */


#include <stdint.h>
#include <immintrin.h>
#include "nrLDPCdecoder_defs.h"
#include "nrLDPC_types.h"
#include "nrLDPC_init.h"
#include "nrLDPC_mPass.h"
#include "nrLDPC_cnProc.h"
#include "nrLDPC_bnProc.h"

#define NR_LDPC_ENABLE_PARITY_CHECK

#ifdef NR_LDPC_DEBUG_MODE
#include "nrLDPC_tools/nrLDPC_debug.h"
#endif

#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include <rte_eal.h>
#include <rte_common.h>
#include <rte_string_fns.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include "nrLDPC_offload.h"

uint8_t count_initdev =0; 
#include <math.h>

#include <rte_dev.h>
#include <rte_launch.h>
#include <rte_bbdev.h>
#include <rte_malloc.h>
#include <rte_random.h>
#include <rte_hexdump.h>
#include <rte_interrupts.h>
//#include "../../../targets/ARCH/test-bbdev/main.h"
#include "../../../targets/ARCH/test-bbdev/test_bbdev_vector.h"


#define GET_SOCKET(socket_id) (((socket_id) == SOCKET_ID_ANY) ? 0 : (socket_id))

#define MAX_QUEUES RTE_MAX_LCORE
#define TEST_REPETITIONS 1

#ifdef RTE_LIBRTE_PMD_BBDEV_FPGA_LTE_FEC
#include <fpga_lte_fec.h>
#define FPGA_LTE_PF_DRIVER_NAME ("intel_fpga_lte_fec_pf")
#define FPGA_LTE_VF_DRIVER_NAME ("intel_fpga_lte_fec_vf")
#define VF_UL_4G_QUEUE_VALUE 4
#define VF_DL_4G_QUEUE_VALUE 4
#define UL_4G_BANDWIDTH 3
#define DL_4G_BANDWIDTH 3
#define UL_4G_LOAD_BALANCE 128
#define DL_4G_LOAD_BALANCE 128
#define FLR_4G_TIMEOUT 610
#endif

#ifdef RTE_LIBRTE_PMD_BBDEV_FPGA_5GNR_FEC
#include <rte_pmd_fpga_5gnr_fec.h>
#define FPGA_5GNR_PF_DRIVER_NAME ("intel_fpga_5gnr_fec_pf")
#define FPGA_5GNR_VF_DRIVER_NAME ("intel_fpga_5gnr_fec_vf")
#define VF_UL_5G_QUEUE_VALUE 4
#define VF_DL_5G_QUEUE_VALUE 4
#define UL_5G_BANDWIDTH 3
#define DL_5G_BANDWIDTH 3
#define UL_5G_LOAD_BALANCE 128
#define DL_5G_LOAD_BALANCE 128
#define FLR_5G_TIMEOUT 610
#endif

#define OPS_CACHE_SIZE 256U
#define OPS_POOL_SIZE_MIN 511U /* 0.5K per queue */

#define SYNC_WAIT 0
#define SYNC_START 1
#define INVALID_OPAQUE -1

#define INVALID_QUEUE_ID -1
/* Increment for next code block in external HARQ memory */
#define HARQ_INCR 32768
/* Headroom for filler LLRs insertion in HARQ buffer */
#define FILLER_HEADROOM 1024

struct test_bbdev_vector test_vector;
struct test_bbdev_vector test_vector_dec;
/* Switch between PMD and Interrupt for throughput TC */
static bool intr_enabled;

/* LLR arithmetic representation for numerical conversion */
static int ldpc_llr_decimals;
static int ldpc_llr_size;
/* Keep track of the LDPC decoder device capability flag */
static uint32_t ldpc_cap_flags;

/* Represents tested active devices */
static struct active_device {
	const char *driver_name;
	uint8_t dev_id;
	uint16_t supported_ops;
	uint16_t queue_ids[MAX_QUEUES];
	uint16_t nb_queues;
	struct rte_mempool *ops_mempool;
	struct rte_mempool *in_mbuf_pool;
	struct rte_mempool *hard_out_mbuf_pool;
	struct rte_mempool *soft_out_mbuf_pool;
	struct rte_mempool *harq_in_mbuf_pool;
	struct rte_mempool *harq_out_mbuf_pool;
} active_devs[RTE_BBDEV_MAX_DEVS];

static uint8_t nb_active_devs;

/* Data buffers used by BBDEV ops */
struct test_buffers {
	struct rte_bbdev_op_data *inputs;
	struct rte_bbdev_op_data *hard_outputs;
	struct rte_bbdev_op_data *soft_outputs;
	struct rte_bbdev_op_data *harq_inputs;
	struct rte_bbdev_op_data *harq_outputs;
};

/* Operation parameters specific for given test case */
struct test_op_params {
	struct rte_mempool *mp;
	struct rte_bbdev_dec_op *ref_dec_op;
	struct rte_bbdev_enc_op *ref_enc_op;
	uint16_t burst_sz;
	uint16_t num_to_process;
	uint16_t num_lcores;
	int vector_mask;
	rte_atomic16_t sync;
	struct test_buffers q_bufs[RTE_MAX_NUMA_NODES][MAX_QUEUES];
};

/* Contains per lcore params */
struct thread_params {
	uint8_t dev_id;
	uint16_t queue_id;
	uint32_t lcore_id;
	uint64_t start_time;
	double ops_per_sec;
	double mbps;
	uint8_t iter_count;
	double iter_average;
	double bler;
	rte_atomic16_t nb_dequeued;
	rte_atomic16_t processing_status;
	rte_atomic16_t burst_sz;
	struct test_op_params *op_params;
	struct rte_bbdev_dec_op *dec_ops[MAX_BURST];
	struct rte_bbdev_enc_op *enc_ops[MAX_BURST];
};

#ifdef RTE_BBDEV_OFFLOAD_COST
/* Stores time statistics */
struct test_time_stats {
	/* Stores software enqueue total working time */
	uint64_t enq_sw_total_time;
	/* Stores minimum value of software enqueue working time */
	uint64_t enq_sw_min_time;
	/* Stores maximum value of software enqueue working time */
	uint64_t enq_sw_max_time;
	/* Stores turbo enqueue total working time */
	uint64_t enq_acc_total_time;
	/* Stores minimum value of accelerator enqueue working time */
	uint64_t enq_acc_min_time;
	/* Stores maximum value of accelerator enqueue working time */
	uint64_t enq_acc_max_time;
	/* Stores dequeue total working time */
	uint64_t deq_total_time;
	/* Stores minimum value of dequeue working time */
	uint64_t deq_min_time;
	/* Stores maximum value of dequeue working time */
	uint64_t deq_max_time;
};
#endif

typedef int (test_case_function)(struct active_device *ad,
		struct test_op_params *op_params);

static inline void
mbuf_reset(struct rte_mbuf *m)
{
	m->pkt_len = 0;

	do {
		m->data_len = 0;
		m = m->next;
	} while (m != NULL);
}

/* Read flag value 0/1 from bitmap */
static inline bool
check_bit(uint32_t bitmap, uint32_t bitmask)
{
	return bitmap & bitmask;
}

static inline void
set_avail_op(struct active_device *ad, enum rte_bbdev_op_type op_type)
{
	ad->supported_ops |= (1 << op_type);
}

static inline bool
is_avail_op(struct active_device *ad, enum rte_bbdev_op_type op_type)
{
	return ad->supported_ops & (1 << op_type);
}

static inline bool
flags_match(uint32_t flags_req, uint32_t flags_present)
{
	return (flags_req & flags_present) == flags_req;
}


static int
check_dev_cap(const struct rte_bbdev_info *dev_info)
{
	unsigned int i;
	const struct rte_bbdev_op_cap *op_cap = dev_info->drv.capabilities;

	for (i = 0; op_cap->type != RTE_BBDEV_OP_NONE; ++i, ++op_cap) {
		if (op_cap->type != RTE_BBDEV_OP_LDPC_DEC)
			continue;

		if (op_cap->type == RTE_BBDEV_OP_LDPC_DEC) {
			const struct rte_bbdev_op_cap_ldpc_dec *cap =
					&op_cap->cap.ldpc_dec;

			if (!flags_match(RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE,
					cap->capability_flags)){
				printf("Flag Mismatch\n");
				return TEST_FAILED;
			}
			return TEST_SUCCESS;
		}
	}

	return TEST_FAILED;
}

/* calculates optimal mempool size not smaller than the val */
static unsigned int
optimal_mempool_size(unsigned int val)
{
	return rte_align32pow2(val + 1) - 1;
}

/* allocates mbuf mempool for inputs and outputs */
static struct rte_mempool *
create_mbuf_pool(uint32_t length, uint8_t dev_id,
		int socket_id, unsigned int mbuf_pool_size,
		const char *op_type_str)
{
	unsigned int i;
	uint32_t max_seg_sz = 0;
	char pool_name[RTE_MEMPOOL_NAMESIZE];

	/* find max input segment size */
	//for (i = 0; i < entries->nb_segments; ++i)
		if (length > max_seg_sz)
			max_seg_sz = length;

	snprintf(pool_name, sizeof(pool_name), "%s_pool_%u", op_type_str,
			dev_id);
	return rte_pktmbuf_pool_create(pool_name, mbuf_pool_size, 0, 0,
			RTE_MAX(max_seg_sz + RTE_PKTMBUF_HEADROOM
					+ FILLER_HEADROOM,
			(unsigned int)RTE_MBUF_DEFAULT_BUF_SIZE), socket_id);
}

static int
create_mempools(struct active_device *ad, int socket_id,
		enum rte_bbdev_op_type org_op_type, uint16_t num_ops, t_nrLDPCoffload_params *p_offloadParams)
{
	struct rte_mempool *mp;
	unsigned int ops_pool_size, mbuf_pool_size = 0;
	char pool_name[RTE_MEMPOOL_NAMESIZE];
	const char *op_type_str;
	enum rte_bbdev_op_type op_type = org_op_type;
	uint8_t nb_segments = 1;  
	uint32_t data_len;
	
	/* allocate ops mempool */
	ops_pool_size = optimal_mempool_size(RTE_MAX(
			/* Ops used plus 1 reference op */
			RTE_MAX((unsigned int)(ad->nb_queues * num_ops + 1),
			/* Minimal cache size plus 1 reference op */
			(unsigned int)(1.5 * rte_lcore_count() *
					OPS_CACHE_SIZE + 1)),
			OPS_POOL_SIZE_MIN));

	if (org_op_type == RTE_BBDEV_OP_NONE)
		op_type = RTE_BBDEV_OP_TURBO_ENC;

	op_type_str = rte_bbdev_op_type_str(op_type);
	TEST_ASSERT_NOT_NULL(op_type_str, "Invalid op type: %u", op_type);

	snprintf(pool_name, sizeof(pool_name), "%s_pool_%u", op_type_str,
			ad->dev_id);
	mp = rte_bbdev_op_pool_create(pool_name, op_type,
			ops_pool_size, OPS_CACHE_SIZE, socket_id);
	TEST_ASSERT_NOT_NULL(mp,
			"ERROR Failed to create %u items ops pool for dev %u on socket %u.",
			ops_pool_size,
			ad->dev_id,
			socket_id);
	ad->ops_mempool = mp;

	/* Do not create inputs and outputs mbufs for BaseBand Null Device */
	if (org_op_type == RTE_BBDEV_OP_NONE)
		return TEST_SUCCESS;
	data_len = (p_offloadParams->BG==1)?(22*p_offloadParams->Z):(10*p_offloadParams->Z);
	/* Inputs */
	if (nb_segments > 0) {
		mbuf_pool_size = optimal_mempool_size(ops_pool_size *
				nb_segments);
		mp = create_mbuf_pool(p_offloadParams->E, ad->dev_id, socket_id,
				mbuf_pool_size, "in");
		TEST_ASSERT_NOT_NULL(mp,
				"ERROR Failed to create %u items input pktmbuf pool for dev %u on socket %u.",
				mbuf_pool_size,
				ad->dev_id,
				socket_id);
		ad->in_mbuf_pool = mp;
	}

	/* Hard outputs */
	if (nb_segments > 0) {
		mbuf_pool_size = optimal_mempool_size(ops_pool_size *
				nb_segments);
		mp = create_mbuf_pool(data_len, ad->dev_id, socket_id,
				mbuf_pool_size,
				"hard_out");
		TEST_ASSERT_NOT_NULL(mp,
				"ERROR Failed to create %u items hard output pktmbuf pool for dev %u on socket %u.",
				mbuf_pool_size,
				ad->dev_id,
				socket_id);
		ad->hard_out_mbuf_pool = mp;
	}

	return TEST_SUCCESS;
}

static int
add_dev(uint8_t dev_id, struct rte_bbdev_info *info)
{
	int ret;
	unsigned int queue_id;
	struct rte_bbdev_queue_conf qconf;
	struct active_device *ad = &active_devs[nb_active_devs];
	unsigned int nb_queues;
	enum rte_bbdev_op_type op_type = RTE_BBDEV_OP_LDPC_DEC; 

/* Configure fpga lte fec with PF & VF values
 * if '-i' flag is set and using fpga device
 */
#ifdef RTE_LIBRTE_PMD_BBDEV_FPGA_LTE_FEC
	if ((get_init_device() == true) &&
		(!strcmp(info->drv.driver_name, FPGA_LTE_PF_DRIVER_NAME))) {
		struct fpga_lte_fec_conf conf;
		unsigned int i;

		printf("Configure FPGA LTE FEC Driver %s with default values\n",
				info->drv.driver_name);

		/* clear default configuration before initialization */
		memset(&conf, 0, sizeof(struct fpga_lte_fec_conf));

		/* Set PF mode :
		 * true if PF is used for data plane
		 * false for VFs
		 */
		conf.pf_mode_en = true;

		for (i = 0; i < FPGA_LTE_FEC_NUM_VFS; ++i) {
			/* Number of UL queues per VF (fpga supports 8 VFs) */
			conf.vf_ul_queues_number[i] = VF_UL_4G_QUEUE_VALUE;
			/* Number of DL queues per VF (fpga supports 8 VFs) */
			conf.vf_dl_queues_number[i] = VF_DL_4G_QUEUE_VALUE;
		}

		/* UL bandwidth. Needed for schedule algorithm */
		conf.ul_bandwidth = UL_4G_BANDWIDTH;
		/* DL bandwidth */
		conf.dl_bandwidth = DL_4G_BANDWIDTH;

		/* UL & DL load Balance Factor to 64 */
		conf.ul_load_balance = UL_4G_LOAD_BALANCE;
		conf.dl_load_balance = DL_4G_LOAD_BALANCE;

		/**< FLR timeout value */
		conf.flr_time_out = FLR_4G_TIMEOUT;

		/* setup FPGA PF with configuration information */
		ret = fpga_lte_fec_configure(info->dev_name, &conf);
		TEST_ASSERT_SUCCESS(ret,
				"Failed to configure 4G FPGA PF for bbdev %s",
				info->dev_name);
	}
#endif
#ifdef RTE_LIBRTE_PMD_BBDEV_FPGA_5GNR_FEC
	if ((get_init_device() == true) &&
		(!strcmp(info->drv.driver_name, FPGA_5GNR_PF_DRIVER_NAME))) {
		struct fpga_5gnr_fec_conf conf;
		unsigned int i;

		printf("Configure FPGA 5GNR FEC Driver %s with default values\n",
				info->drv.driver_name);

		/* clear default configuration before initialization */
		memset(&conf, 0, sizeof(struct fpga_5gnr_fec_conf));

		/* Set PF mode :
		 * true if PF is used for data plane
		 * false for VFs
		 */
		conf.pf_mode_en = true;

		for (i = 0; i < FPGA_5GNR_FEC_NUM_VFS; ++i) {
			/* Number of UL queues per VF (fpga supports 8 VFs) */
			conf.vf_ul_queues_number[i] = VF_UL_5G_QUEUE_VALUE;
			/* Number of DL queues per VF (fpga supports 8 VFs) */
			conf.vf_dl_queues_number[i] = VF_DL_5G_QUEUE_VALUE;
		}

		/* UL bandwidth. Needed for schedule algorithm */
		conf.ul_bandwidth = UL_5G_BANDWIDTH;
		/* DL bandwidth */
		conf.dl_bandwidth = DL_5G_BANDWIDTH;

		/* UL & DL load Balance Factor to 64 */
		conf.ul_load_balance = UL_5G_LOAD_BALANCE;
		conf.dl_load_balance = DL_5G_LOAD_BALANCE;

		/**< FLR timeout value */
		conf.flr_time_out = FLR_5G_TIMEOUT;

		/* setup FPGA PF with configuration information */
		ret = fpga_5gnr_fec_configure(info->dev_name, &conf);
		TEST_ASSERT_SUCCESS(ret,
				"Failed to configure 5G FPGA PF for bbdev %s",
				info->dev_name);
	}
#endif
	nb_queues = RTE_MIN(rte_lcore_count(), info->drv.max_num_queues);
	nb_queues = RTE_MIN(nb_queues, (unsigned int) MAX_QUEUES);

	/* setup device */
	ret = rte_bbdev_setup_queues(dev_id, nb_queues, info->socket_id);
	if (ret < 0) {
		printf("rte_bbdev_setup_queues(%u, %u, %d) ret %i\n",
				dev_id, nb_queues, info->socket_id, ret);
		return TEST_FAILED;
	}

	/* configure interrupts if needed */
	if (intr_enabled) {
		ret = rte_bbdev_intr_enable(dev_id);
		if (ret < 0) {
			printf("rte_bbdev_intr_enable(%u) ret %i\n", dev_id,
					ret);
			return TEST_FAILED;
		}
	}

	/* setup device queues */
	qconf.socket = info->socket_id;
	qconf.queue_size = info->drv.default_queue_conf.queue_size;
	qconf.priority = 0;
	qconf.deferred_start = 0;
	qconf.op_type = op_type;

	for (queue_id = 0; queue_id < nb_queues; ++queue_id) {
		ret = rte_bbdev_queue_configure(dev_id, queue_id, &qconf);
		if (ret != 0) {
			printf(
					"Allocated all queues (id=%u) at prio%u on dev%u\n",
					queue_id, qconf.priority, dev_id);
			qconf.priority++;
			ret = rte_bbdev_queue_configure(ad->dev_id, queue_id,
					&qconf);
		}
		if (ret != 0) {
			printf("All queues on dev %u allocated: %u\n",
					dev_id, queue_id);
			break;
		}
		ad->queue_ids[queue_id] = queue_id;
	}
	TEST_ASSERT(queue_id != 0,
			"ERROR Failed to configure any queues on dev %u",
			dev_id);
	ad->nb_queues = queue_id;

	set_avail_op(ad, op_type);

	return TEST_SUCCESS;
}

static int
add_active_device(uint8_t dev_id, struct rte_bbdev_info *info)
{
	int ret;

	active_devs[nb_active_devs].driver_name = info->drv.driver_name;
	active_devs[nb_active_devs].dev_id = dev_id;

	ret = add_dev(dev_id, info);
	if (ret == TEST_SUCCESS)
		++nb_active_devs;
	return ret;
}

static uint8_t
populate_active_devices(void)
{
	int ret;
	uint8_t dev_id;
	uint8_t nb_devs_added = 0;
	struct rte_bbdev_info info;

	RTE_BBDEV_FOREACH(dev_id) {
		rte_bbdev_info_get(dev_id, &info);

		if (check_dev_cap(&info)) {
			printf(
				"Device %d (%s) does not support specified capabilities\n",
					dev_id, info.dev_name);
			continue;
		}
		ret = add_active_device(dev_id, &info);
		if (ret != 0) {
			printf("Adding active bbdev %s skipped\n",
					info.dev_name);
			continue;
		}
		nb_devs_added++;
	}

	return nb_devs_added;
}

static int
device_setup(void)
{
if (count_initdev ==0) {
count_initdev++;
	if (populate_active_devices() == 0) {
		printf("No suitable devices found!\n");
		return TEST_SKIPPED;
	}
}
	return TEST_SUCCESS;
}


static void
testsuite_teardown(void)
{
	uint8_t dev_id;
printf("testsuite teardown \n"); 
	/* Unconfigure devices */
//	RTE_BBDEV_FOREACH(dev_id)
//		rte_bbdev_close(dev_id);

	/* Clear active devices structs. */
//	memset(active_devs, 0, sizeof(active_devs));
//	nb_active_devs = 0;

	/* Disable interrupts */
//	intr_enabled = false;
}

static int
ut_setup(void)
{
	uint8_t i, dev_id;

	for (i = 0; i < nb_active_devs; i++) {
		dev_id = active_devs[i].dev_id;
		/* reset bbdev stats */
		TEST_ASSERT_SUCCESS(rte_bbdev_stats_reset(dev_id),
				"Failed to reset stats of bbdev %u", dev_id);
		/* start the device */
		TEST_ASSERT_SUCCESS(rte_bbdev_start(dev_id),
				"Failed to start bbdev %u", dev_id);
	}

	return TEST_SUCCESS;
}

static void
ut_teardown(void)
{
	uint8_t i, dev_id;
	struct rte_bbdev_stats stats;

	for (i = 0; i < nb_active_devs; i++) {
		dev_id = active_devs[i].dev_id;
		/* read stats and print */
		rte_bbdev_stats_get(dev_id, &stats);
		/* Stop the device */
		//rte_bbdev_stop(dev_id);
	}
}

static int
init_op_data_objs(struct rte_bbdev_op_data *bufs, 
		int8_t* p_llr, uint32_t data_len,
		struct rte_mempool *mbuf_pool, const uint16_t n,
		enum op_data_type op_type, uint16_t min_alignment)
{
	int ret;
	unsigned int i, j;
	bool large_input = false;
	uint8_t nb_segments = 1;

	for (i = 0; i < n; ++i) {
		char *data;
		struct rte_mbuf *m_head = rte_pktmbuf_alloc(mbuf_pool);
		TEST_ASSERT_NOT_NULL(m_head,
				"Not enough mbufs in %d data type mbuf pool (needed %u, available %u)",
				op_type, n * nb_segments,
				mbuf_pool->size);

		if (data_len > RTE_BBDEV_LDPC_E_MAX_MBUF) {
			/*
			 * Special case when DPDK mbuf cannot handle
			 * the required input size
			 */
			printf("Warning: Larger input size than DPDK mbuf %d\n",
					data_len);
			large_input = true;
		}
		bufs[i].data = m_head;
		bufs[i].offset = 0;
		bufs[i].length = 0;

		if ((op_type == DATA_INPUT) || (op_type == DATA_HARQ_INPUT)) {
			if ((op_type == DATA_INPUT) && large_input) {
				/* Allocate a fake overused mbuf */
				data = rte_malloc(NULL, data_len, 0);
				memcpy(data, p_llr, data_len);
				m_head->buf_addr = data;
				m_head->buf_iova = rte_malloc_virt2iova(data);
				m_head->data_off = 0;
				m_head->data_len = data_len;
			} else {
				data = rte_pktmbuf_append(m_head, data_len);
				TEST_ASSERT_NOT_NULL(data,
					"Couldn't append %u bytes to mbuf from %d data type mbuf pool",
					data_len, op_type);

				TEST_ASSERT(data == RTE_PTR_ALIGN(
						data, min_alignment),
					"Data addr in mbuf (%p) is not aligned to device min alignment (%u)",
					data, min_alignment);
				rte_memcpy(data, p_llr, data_len);
			}

			bufs[i].length += data_len;

			for (j = 1; j < nb_segments; ++j) {
				struct rte_mbuf *m_tail =
						rte_pktmbuf_alloc(mbuf_pool);
				TEST_ASSERT_NOT_NULL(m_tail,
						"Not enough mbufs in %d data type mbuf pool (needed %u, available %u)",
						op_type,
						n * nb_segments,
						mbuf_pool->size);
				//seg += 1;

				data = rte_pktmbuf_append(m_tail, data_len);
				TEST_ASSERT_NOT_NULL(data,
						"Couldn't append %u bytes to mbuf from %d data type mbuf pool",
						data_len, op_type);

				TEST_ASSERT(data == RTE_PTR_ALIGN(data,
						min_alignment),
						"Data addr in mbuf (%p) is not aligned to device min alignment (%u)",
						data, min_alignment);
				rte_memcpy(data, p_llr, data_len);
				bufs[i].length += data_len;

				ret = rte_pktmbuf_chain(m_head, m_tail);
				TEST_ASSERT_SUCCESS(ret,
						"Couldn't chain mbufs from %d data type mbuf pool",
						op_type);
			}
		} else {

			/* allocate chained-mbuf for output buffer */
			for (j = 1; j < nb_segments; ++j) {
				struct rte_mbuf *m_tail =
						rte_pktmbuf_alloc(mbuf_pool);
				TEST_ASSERT_NOT_NULL(m_tail,
						"Not enough mbufs in %d data type mbuf pool (needed %u, available %u)",
						op_type,
						n * nb_segments,
						mbuf_pool->size);

				ret = rte_pktmbuf_chain(m_head, m_tail);
				TEST_ASSERT_SUCCESS(ret,
						"Couldn't chain mbufs from %d data type mbuf pool",
						op_type);
			}
		}
	}

	return 0;
}

static int
allocate_buffers_on_socket(struct rte_bbdev_op_data **buffers, const int len,
		const int socket)
{
	int i;

	*buffers = rte_zmalloc_socket(NULL, len, 0, socket);
	if (*buffers == NULL) {
		printf("WARNING: Failed to allocate op_data on socket %d\n",
				socket);
		/* try to allocate memory on other detected sockets */
		for (i = 0; i < socket; i++) {
			*buffers = rte_zmalloc_socket(NULL, len, 0, i);
			if (*buffers != NULL)
				break;
		}
	}

	return (*buffers == NULL) ? TEST_FAILED : TEST_SUCCESS;
}


/*
 * We may have to insert filler bits
 * when they are required by the HARQ assumption
 */
static void
ldpc_add_filler(struct rte_bbdev_op_data *input_ops,
		const uint16_t n, struct test_op_params *op_params)
{
	struct rte_bbdev_op_ldpc_dec dec = op_params->ref_dec_op->ldpc_dec;

	if (input_ops == NULL)
		return;
	/* No need to add filler if not required by device */
	if (!(ldpc_cap_flags &
			RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_FILLERS))
		return;
	/* No need to add filler for loopback operation */
	if (dec.op_flags & RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_LOOPBACK)
		return;

	uint16_t i, j, parity_offset;
	for (i = 0; i < n; ++i) {
		struct rte_mbuf *m = input_ops[i].data;
		int8_t *llr = rte_pktmbuf_mtod_offset(m, int8_t *,
				input_ops[i].offset);
		parity_offset = (dec.basegraph == 1 ? 20 : 8)
				* dec.z_c - dec.n_filler;
		uint16_t new_hin_size = input_ops[i].length + dec.n_filler;
		m->data_len = new_hin_size;
		input_ops[i].length = new_hin_size;
		for (j = new_hin_size - 1; j >= parity_offset + dec.n_filler;
				j--)
			llr[j] = llr[j - dec.n_filler];
		uint16_t llr_max_pre_scaling = (1 << (ldpc_llr_size - 1)) - 1;
		for (j = 0; j < dec.n_filler; j++)
			llr[parity_offset + j] = llr_max_pre_scaling;
	}
}

static void
ldpc_input_llr_scaling(struct rte_bbdev_op_data *input_ops,
		const uint16_t n, const int8_t llr_size,
		const int8_t llr_decimals)
{
	if (input_ops == NULL)
		return;

	uint16_t i, byte_idx;

	int16_t llr_max, llr_min, llr_tmp;
	llr_max = (1 << (llr_size - 1)) - 1;
	llr_min = -llr_max;
	for (i = 0; i < n; ++i) {
		struct rte_mbuf *m = input_ops[i].data;
		while (m != NULL) {
			int8_t *llr = rte_pktmbuf_mtod_offset(m, int8_t *,
					input_ops[i].offset);
			for (byte_idx = 0; byte_idx < rte_pktmbuf_data_len(m);
					++byte_idx) {

				llr_tmp = llr[byte_idx];
				if (llr_decimals == 4)
					llr_tmp *= 8;
				else if (llr_decimals == 2)
					llr_tmp *= 2;
				else if (llr_decimals == 0)
					llr_tmp /= 2;
				llr_tmp = RTE_MIN(llr_max,
						RTE_MAX(llr_min, llr_tmp));
				llr[byte_idx] = (int8_t) llr_tmp;
			}

			m = m->next;
		}
	}
}



static int
fill_queue_buffers(struct test_op_params *op_params,int8_t* p_llr, uint32_t data_len,
		struct rte_mempool *in_mp, struct rte_mempool *hard_out_mp,
		struct rte_mempool *soft_out_mp,
		struct rte_mempool *harq_in_mp, struct rte_mempool *harq_out_mp,
		uint16_t queue_id,
		const struct rte_bbdev_op_cap *capabilities,
		uint16_t min_alignment, const int socket_id)
{
	int ret;
	enum op_data_type type;
	const uint16_t n = op_params->num_to_process;

	struct rte_mempool *mbuf_pools[DATA_NUM_TYPES] = {
		in_mp,
		soft_out_mp,
		hard_out_mp,
		harq_in_mp,
		harq_out_mp,
	};

	struct rte_bbdev_op_data **queue_ops[DATA_NUM_TYPES] = {
		&op_params->q_bufs[socket_id][queue_id].inputs,
		&op_params->q_bufs[socket_id][queue_id].soft_outputs,
		&op_params->q_bufs[socket_id][queue_id].hard_outputs,
		&op_params->q_bufs[socket_id][queue_id].harq_inputs,
		&op_params->q_bufs[socket_id][queue_id].harq_outputs,
	};

	for (type = DATA_INPUT; type < 3; type+=2) {

		ret = allocate_buffers_on_socket(queue_ops[type],
				n * sizeof(struct rte_bbdev_op_data),
				socket_id);
		TEST_ASSERT_SUCCESS(ret,
				"Couldn't allocate memory for rte_bbdev_op_data structs");

		ret = init_op_data_objs(*queue_ops[type], p_llr, data_len,
				mbuf_pools[type], n, type, min_alignment);
		TEST_ASSERT_SUCCESS(ret,
				"Couldn't init rte_bbdev_op_data structs");
	}

//	if (test_vector.op_type == RTE_BBDEV_OP_LDPC_DEC) {
		bool loopback = op_params->ref_dec_op->ldpc_dec.op_flags &
				RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_LOOPBACK;
		bool llr_comp = op_params->ref_dec_op->ldpc_dec.op_flags &
				RTE_BBDEV_LDPC_LLR_COMPRESSION;
		bool harq_comp = op_params->ref_dec_op->ldpc_dec.op_flags &
				RTE_BBDEV_LDPC_HARQ_6BIT_COMPRESSION;
		ldpc_llr_decimals = capabilities->cap.ldpc_dec.llr_decimals;
		ldpc_llr_size = capabilities->cap.ldpc_dec.llr_size;
		ldpc_cap_flags = capabilities->cap.ldpc_dec.capability_flags;
		if (!loopback && !llr_comp)
			ldpc_input_llr_scaling(*queue_ops[DATA_INPUT], n,
					ldpc_llr_size, ldpc_llr_decimals);
		if (!loopback && !harq_comp)
			ldpc_input_llr_scaling(*queue_ops[DATA_HARQ_INPUT], n,
					ldpc_llr_size, ldpc_llr_decimals);
		if (!loopback)
			ldpc_add_filler(*queue_ops[DATA_HARQ_INPUT], n,
					op_params);
//	}

	return 0;
}

static void
free_buffers(struct active_device *ad, struct test_op_params *op_params)
{
	unsigned int i, j;

	rte_mempool_free(ad->ops_mempool);
	rte_mempool_free(ad->in_mbuf_pool);
	rte_mempool_free(ad->hard_out_mbuf_pool);
	rte_mempool_free(ad->soft_out_mbuf_pool);
	rte_mempool_free(ad->harq_in_mbuf_pool);
	rte_mempool_free(ad->harq_out_mbuf_pool);

	for (i = 0; i < rte_lcore_count(); ++i) {
		for (j = 0; j < RTE_MAX_NUMA_NODES; ++j) {
			rte_free(op_params->q_bufs[j][i].inputs);
			rte_free(op_params->q_bufs[j][i].hard_outputs);
			rte_free(op_params->q_bufs[j][i].soft_outputs);
			rte_free(op_params->q_bufs[j][i].harq_inputs);
			rte_free(op_params->q_bufs[j][i].harq_outputs);
		}
	}
}



static inline double
maxstar(double A, double B)
{
	if (fabs(A - B) > 5)
		return RTE_MAX(A, B);
	else
		return RTE_MAX(A, B) + log1p(exp(-fabs(A - B)));
}

static void
copy_reference_ldpc_dec_op(struct rte_bbdev_dec_op **ops, unsigned int n,
		unsigned int start_idx,
		struct rte_bbdev_op_data *inputs,
		struct rte_bbdev_op_data *hard_outputs,
		struct rte_bbdev_op_data *soft_outputs,
		struct rte_bbdev_op_data *harq_inputs,
		struct rte_bbdev_op_data *harq_outputs,
		struct rte_bbdev_dec_op *ref_op, 
		t_nrLDPCoffload_params *p_offloadParams)
{
	unsigned int i;
	struct rte_bbdev_op_ldpc_dec *ldpc_dec = &ref_op->ldpc_dec;
	for (i = 0; i < n; ++i) {
	/*	if (ldpc_dec->code_block_mode == 0) {
			ops[i]->ldpc_dec.tb_params.ea =
					ldpc_dec->tb_params.ea;
			ops[i]->ldpc_dec.tb_params.eb =
					ldpc_dec->tb_params.eb;
			ops[i]->ldpc_dec.tb_params.c =
					ldpc_dec->tb_params.c;
			ops[i]->ldpc_dec.tb_params.cab =
					ldpc_dec->tb_params.cab;
			ops[i]->ldpc_dec.tb_params.r =
					ldpc_dec->tb_params.r;
					printf("code block ea %d eb %d c %d cab %d r %d\n",ldpc_dec->tb_params.ea,ldpc_dec->tb_params.eb,ldpc_dec->tb_params.c, ldpc_dec->tb_params.cab, ldpc_dec->tb_params.r);
		} else { */
			ops[i]->ldpc_dec.cb_params.e = p_offloadParams->E; 
		//}

		ops[i]->ldpc_dec.basegraph = p_offloadParams->BG; 
		ops[i]->ldpc_dec.z_c = p_offloadParams->Z; 
		ops[i]->ldpc_dec.q_m = p_offloadParams->Qm; 
		ops[i]->ldpc_dec.n_filler = p_offloadParams->F; 
		ops[i]->ldpc_dec.n_cb = p_offloadParams->n_cb;
		ops[i]->ldpc_dec.iter_max = 8; 
		ops[i]->ldpc_dec.rv_index = p_offloadParams->rv; 
		ops[i]->ldpc_dec.op_flags = RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
		ops[i]->ldpc_dec.code_block_mode = 1; //ldpc_dec->code_block_mode;
		

		if (hard_outputs != NULL)
			ops[i]->ldpc_dec.hard_output =
					hard_outputs[start_idx + i];
		if (inputs != NULL)
			ops[i]->ldpc_dec.input =
					inputs[start_idx + i];
		if (soft_outputs != NULL)
			ops[i]->ldpc_dec.soft_output =
					soft_outputs[start_idx + i];
		if (harq_inputs != NULL)
			ops[i]->ldpc_dec.harq_combined_input =
					harq_inputs[start_idx + i];
		if (harq_outputs != NULL)
			ops[i]->ldpc_dec.harq_combined_output =
					harq_outputs[start_idx + i];

	}
}



static int
check_dec_status_and_ordering(struct rte_bbdev_dec_op *op,
		unsigned int order_idx, const int expected_status)
{
	int status = op->status;
	/* ignore parity mismatch false alarms for long iterations */
	if (get_iter_max() >= 10) {
		if (!(expected_status & (1 << RTE_BBDEV_SYNDROME_ERROR)) &&
				(status & (1 << RTE_BBDEV_SYNDROME_ERROR))) {
			printf("WARNING: Ignore Syndrome Check mismatch\n");
			status -= (1 << RTE_BBDEV_SYNDROME_ERROR);
		}
		if ((expected_status & (1 << RTE_BBDEV_SYNDROME_ERROR)) &&
				!(status & (1 << RTE_BBDEV_SYNDROME_ERROR))) {
			printf("WARNING: Ignore Syndrome Check mismatch\n");
			status += (1 << RTE_BBDEV_SYNDROME_ERROR);
		}
	}

	TEST_ASSERT(status == expected_status,
			"op_status (%d) != expected_status (%d)",
			op->status, expected_status);

	TEST_ASSERT((void *)(uintptr_t)order_idx == op->opaque_data,
			"Ordering error, expected %p, got %p",
			(void *)(uintptr_t)order_idx, op->opaque_data);

	return TEST_SUCCESS;
}

/*static int
check_enc_status_and_ordering(struct rte_bbdev_enc_op *op,
		unsigned int order_idx, const int expected_status)
{
	TEST_ASSERT(op->status == expected_status,
			"op_status (%d) != expected_status (%d)",
			op->status, expected_status);

	if (op->opaque_data != (void *)(uintptr_t)INVALID_OPAQUE)
		TEST_ASSERT((void *)(uintptr_t)order_idx == op->opaque_data,
				"Ordering error, expected %p, got %p",
				(void *)(uintptr_t)order_idx, op->opaque_data);

	return TEST_SUCCESS;
}
*/

static int
validate_ldpc_dec_op(struct rte_bbdev_dec_op **ops, const uint16_t n,
		struct rte_bbdev_dec_op *ref_op, const int vector_mask,
		int8_t* p_out)
{
	unsigned int i;
	int ret;
	struct op_data_entries *hard_data_orig ; 
	struct rte_bbdev_op_ldpc_dec *ops_td;
	struct rte_bbdev_op_data *hard_output;
	struct rte_bbdev_op_data *harq_output;
	struct rte_bbdev_op_data *soft_output;
	struct rte_bbdev_op_ldpc_dec *ref_td = &ref_op->ldpc_dec;
	struct rte_mbuf *m;  
	char *data;

	for (i = 0; i < n; ++i) {
		ops_td = &ops[i]->ldpc_dec;
		hard_output = &ops_td->hard_output;
		m = hard_output->data;
	/*	ret = check_dec_status_and_ordering(ops[i], i, ref_op->status);
		TEST_ASSERT_SUCCESS(ret,
				"Checking status and ordering for decoder failed");
		if (vector_mask & TEST_BBDEV_VF_EXPECTED_ITER_COUNT)
			TEST_ASSERT(ops_td->iter_count <= ref_td->iter_count,
					"Returned iter_count (%d) > expected iter_count (%d)",
					ops_td->iter_count, ref_td->iter_count);
*/
		uint16_t offset = hard_output->offset;
                uint16_t data_len = rte_pktmbuf_data_len(m) - offset;

                data = m->buf_addr;
                memcpy(p_out, data+m->data_off, data_len);

	}

	return TEST_SUCCESS;
}


/*static int
validate_ldpc_enc_op(struct rte_bbdev_enc_op **ops, const uint16_t n,
		struct rte_bbdev_enc_op *ref_op)
{
	unsigned int i;
	int ret;
	struct op_data_entries *hard_data_orig =
			&test_vector.entries[DATA_HARD_OUTPUT];

	for (i = 0; i < n; ++i) {
		ret = check_enc_status_and_ordering(ops[i], i, ref_op->status);
		TEST_ASSERT_SUCCESS(ret,
				"Checking status and ordering for encoder failed");
		TEST_ASSERT_SUCCESS(validate_op_chain(
				&ops[i]->ldpc_enc.output,
				hard_data_orig),
				"Output buffers (CB=%u) are not equal",
				i);
	}

	return TEST_SUCCESS;
}
*/

static void
create_reference_ldpc_dec_op(struct rte_bbdev_dec_op *op, t_nrLDPCoffload_params *p_offloadParams)
{
//	unsigned int i;
//	struct op_data_entries *entry;

//	op->ldpc_dec = test_vector.ldpc_dec;
//	entry = &test_vector.entries[DATA_INPUT];
	//for (i = 0; i < entry->nb_segments; ++i)
		op->ldpc_dec.input.length = p_offloadParams->E; 
//				entry->segments[i].length;
	/*if (test_vector.ldpc_dec.op_flags &
			RTE_BBDEV_LDPC_HQ_COMBINE_IN_ENABLE) {
		entry = &test_vector.entries[DATA_HARQ_INPUT];
		for (i = 0; i < entry->nb_segments; ++i)
			op->ldpc_dec.harq_combined_input.length +=
				entry->segments[i].length;
	}*/
}




static uint32_t
calc_ldpc_dec_TB_size(struct rte_bbdev_dec_op *op)
{
	uint8_t i;
	uint32_t c, r, tb_size = 0;
	uint16_t sys_cols = (op->ldpc_dec.basegraph == 1) ? 22 : 10;

	if (op->ldpc_dec.code_block_mode) {
		tb_size = sys_cols * op->ldpc_dec.z_c - op->ldpc_dec.n_filler;
	} else {
		c = op->ldpc_dec.tb_params.c;
		r = op->ldpc_dec.tb_params.r;
		for (i = 0; i < c-r; i++)
			tb_size += sys_cols * op->ldpc_dec.z_c
					- op->ldpc_dec.n_filler;
					printf("calc tb c %d r %d sys cols %d tb_size %d\n",c,r,sys_cols,tb_size);
	}
	return tb_size;
}


static int
init_test_op_params(struct test_op_params *op_params,
		enum rte_bbdev_op_type op_type, const int expected_status,
		const int vector_mask, struct rte_mempool *ops_mp,
		uint16_t burst_sz, uint16_t num_to_process, uint16_t num_lcores)
{
	int ret = 0;
	if (op_type == RTE_BBDEV_OP_TURBO_DEC ||
			op_type == RTE_BBDEV_OP_LDPC_DEC)
		ret = rte_bbdev_dec_op_alloc_bulk(ops_mp,
				&op_params->ref_dec_op, 1);
	else
		ret = rte_bbdev_enc_op_alloc_bulk(ops_mp,
				&op_params->ref_enc_op, 1);

	TEST_ASSERT_SUCCESS(ret, "rte_bbdev_op_alloc_bulk() failed");

	op_params->mp = ops_mp;
	op_params->burst_sz = burst_sz;
	op_params->num_to_process = num_to_process;
	op_params->num_lcores = num_lcores;
	op_params->vector_mask = vector_mask;
	if (op_type == RTE_BBDEV_OP_TURBO_DEC ||
			op_type == RTE_BBDEV_OP_LDPC_DEC)
		op_params->ref_dec_op->status = expected_status;
	else if (op_type == RTE_BBDEV_OP_TURBO_ENC
			|| op_type == RTE_BBDEV_OP_LDPC_ENC)
		op_params->ref_enc_op->status = expected_status;
	return 0;
}

static int
decod_on_device(uint8_t dev_id,
		struct test_op_params *op_params, int8_t* p_llr, 
		t_nrLDPCoffload_params *p_offloadParams, int8_t* p_out) 
{

	int t_ret, f_ret, socket_id = SOCKET_ID_ANY;
	unsigned int i;
	struct active_device *ad;
	unsigned int burst_sz = get_burst_sz();
	enum rte_bbdev_op_type op_type = RTE_BBDEV_OP_LDPC_DEC;
	const struct rte_bbdev_op_cap *capabilities = NULL;

	ad = &active_devs[dev_id];

	/* Check if device supports op_type */
	if (!is_avail_op(ad, op_type))
		return TEST_SUCCESS;

	struct rte_bbdev_info info;
	rte_bbdev_info_get(ad->dev_id, &info);
	socket_id = GET_SOCKET(info.socket_id);

	f_ret = create_mempools(ad, socket_id, op_type,
			get_num_ops(),p_offloadParams);
	if (f_ret != TEST_SUCCESS) {
		printf("Couldn't create mempools");
		goto fail;
	}
	f_ret = init_test_op_params(op_params, op_type,
			0,
			0,
			ad->ops_mempool,
			burst_sz,
			get_num_ops(),
			get_num_lcores());
	if (f_ret != TEST_SUCCESS) {
		printf("Couldn't init test op params");
		goto fail;
	}


	/* Find capabilities */
	const struct rte_bbdev_op_cap *cap = info.drv.capabilities;
	for (i = 0; i < RTE_BBDEV_OP_TYPE_COUNT; i++) {
		if (cap->type == op_type) {
			capabilities = cap;
			break;
		}
		cap++;
	}
	TEST_ASSERT_NOT_NULL(capabilities,
			"Couldn't find capabilities");

	create_reference_ldpc_dec_op(op_params->ref_dec_op, p_offloadParams);

	for (i = 0; i < ad->nb_queues; ++i) {
		f_ret = fill_queue_buffers(op_params,
			 	p_llr,	
			 	p_offloadParams->E,	
				ad->in_mbuf_pool,
				ad->hard_out_mbuf_pool,
				ad->soft_out_mbuf_pool,
				ad->harq_in_mbuf_pool,
				ad->harq_out_mbuf_pool,
				ad->queue_ids[i],
				capabilities,
				info.drv.min_alignment,
				socket_id);
		if (f_ret != TEST_SUCCESS) {
			printf("Couldn't init queue buffers");
			goto fail;
		}
	}

	t_ret = start_pmd_dec(ad, op_params, p_offloadParams, p_out);

	/* Free active device resources and return */
	free_buffers(ad, op_params);
	return t_ret;

fail:
	free_buffers(ad, op_params);
	return TEST_FAILED;
}

/* Run ldpc per active device per supported op type
 * per burst size.
 */
static int
ldpc_decod_ut(int8_t* p_llr, t_nrLDPCoffload_params *p_offloadParams, int8_t* p_out) 
{
	int ret = 0;
	uint8_t dev;

	/* Alloc op_params */
	struct test_op_params *op_params = rte_zmalloc(NULL,
			sizeof(struct test_op_params), RTE_CACHE_LINE_SIZE);
	TEST_ASSERT_NOT_NULL(op_params, "Failed to alloc %zuB for op_params",
			RTE_ALIGN(sizeof(struct test_op_params),
				RTE_CACHE_LINE_SIZE));
	/* For each device run test case function */
	for (dev = 0; dev < nb_active_devs; ++dev)
		ret |= decod_on_device(dev, op_params, p_llr, p_offloadParams, p_out);

	rte_free(op_params);

	return ret;
}


/* Push back the HARQ output from DDR to host */
static void
retrieve_harq_ddr(uint16_t dev_id, uint16_t queue_id,
		struct rte_bbdev_dec_op **ops,
		const uint16_t n)
{
	uint16_t j;
	int save_status, ret;
	uint32_t harq_offset = (uint32_t) queue_id * HARQ_INCR * 1024;
	struct rte_bbdev_dec_op *ops_deq[MAX_BURST];
	uint32_t flags = ops[0]->ldpc_dec.op_flags;
	bool loopback = flags & RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_LOOPBACK;
	bool mem_out = flags & RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_OUT_ENABLE;
	bool hc_out = flags & RTE_BBDEV_LDPC_HQ_COMBINE_OUT_ENABLE;
	bool h_comp = flags & RTE_BBDEV_LDPC_HARQ_6BIT_COMPRESSION;
	for (j = 0; j < n; ++j) {
		if ((loopback && mem_out) || hc_out) {
			save_status = ops[j]->status;
			ops[j]->ldpc_dec.op_flags =
				RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_LOOPBACK +
				RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_IN_ENABLE;
			if (h_comp)
				ops[j]->ldpc_dec.op_flags +=
					RTE_BBDEV_LDPC_HARQ_6BIT_COMPRESSION;
			ops[j]->ldpc_dec.harq_combined_input.offset =
					harq_offset;
			ops[j]->ldpc_dec.harq_combined_output.offset = 0;
			harq_offset += HARQ_INCR;
			if (!loopback)
				ops[j]->ldpc_dec.harq_combined_input.length =
				ops[j]->ldpc_dec.harq_combined_output.length;
			rte_bbdev_enqueue_ldpc_dec_ops(dev_id, queue_id,
					&ops[j], 1);
			ret = 0;
			while (ret == 0)
				ret = rte_bbdev_dequeue_ldpc_dec_ops(
						dev_id, queue_id,
						&ops_deq[j], 1);
			ops[j]->ldpc_dec.op_flags = flags;
			ops[j]->status = save_status;
		}
	}
}

/*
 * Push back the HARQ output from HW DDR to Host
 * Preload HARQ memory input and adjust HARQ offset
 */
static void
preload_harq_ddr(uint16_t dev_id, uint16_t queue_id,
		struct rte_bbdev_dec_op **ops, const uint16_t n,
		bool preload)
{
	uint16_t j;
	int ret;
	uint32_t harq_offset = (uint32_t) queue_id * HARQ_INCR * 1024;
	struct rte_bbdev_op_data save_hc_in, save_hc_out;
	struct rte_bbdev_dec_op *ops_deq[MAX_BURST];
	uint32_t flags = ops[0]->ldpc_dec.op_flags;
	bool mem_in = flags & RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_IN_ENABLE;
	bool hc_in = flags & RTE_BBDEV_LDPC_HQ_COMBINE_IN_ENABLE;
	bool mem_out = flags & RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_OUT_ENABLE;
	bool hc_out = flags & RTE_BBDEV_LDPC_HQ_COMBINE_OUT_ENABLE;
	bool h_comp = flags & RTE_BBDEV_LDPC_HARQ_6BIT_COMPRESSION;
	for (j = 0; j < n; ++j) {
		if ((mem_in || hc_in) && preload) {
			save_hc_in = ops[j]->ldpc_dec.harq_combined_input;
			save_hc_out = ops[j]->ldpc_dec.harq_combined_output;
			ops[j]->ldpc_dec.op_flags =
				RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_LOOPBACK +
				RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_OUT_ENABLE;
			if (h_comp)
				ops[j]->ldpc_dec.op_flags +=
					RTE_BBDEV_LDPC_HARQ_6BIT_COMPRESSION;
			ops[j]->ldpc_dec.harq_combined_output.offset =
					harq_offset;
			ops[j]->ldpc_dec.harq_combined_input.offset = 0;
			rte_bbdev_enqueue_ldpc_dec_ops(dev_id, queue_id,
					&ops[j], 1);
			ret = 0;
			while (ret == 0)
				ret = rte_bbdev_dequeue_ldpc_dec_ops(
					dev_id, queue_id, &ops_deq[j], 1);
			ops[j]->ldpc_dec.op_flags = flags;
			ops[j]->ldpc_dec.harq_combined_input = save_hc_in;
			ops[j]->ldpc_dec.harq_combined_output = save_hc_out;
		}
		/* Adjust HARQ offset when we reach external DDR */
		if (mem_in || hc_in)
			ops[j]->ldpc_dec.harq_combined_input.offset
				= harq_offset;
		if (mem_out || hc_out)
			ops[j]->ldpc_dec.harq_combined_output.offset
				= harq_offset;
		harq_offset += HARQ_INCR;
	}
}


static int
pmd_lcore_ldpc_dec(void *arg, t_nrLDPCoffload_params *p_offloadParams, int8_t* p_out)
{
	struct thread_params *tp = arg;
	uint16_t enq, deq;
	uint64_t total_time = 0, start_time;
	const uint16_t queue_id = tp->queue_id;
	const uint16_t burst_sz = tp->op_params->burst_sz;
	const uint16_t num_ops = tp->op_params->num_to_process;
	struct rte_bbdev_dec_op *ops_enq[num_ops];
	struct rte_bbdev_dec_op *ops_deq[num_ops];
	struct rte_bbdev_dec_op *ref_op = tp->op_params->ref_dec_op;
	struct test_buffers *bufs = NULL;
	int i, j, ret;
	struct rte_bbdev_info info;
	uint16_t num_to_enq;

        struct rte_bbdev_op_data *hard_output;	
        struct rte_bbdev_op_ldpc_dec *ops_td;
       
        bool extDdr = check_bit(ldpc_cap_flags,
			RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_OUT_ENABLE);
	bool loopback = check_bit(ref_op->ldpc_dec.op_flags,
			RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_LOOPBACK);
	bool hc_out = check_bit(ref_op->ldpc_dec.op_flags,
			RTE_BBDEV_LDPC_HQ_COMBINE_OUT_ENABLE);

	TEST_ASSERT_SUCCESS((burst_sz > MAX_BURST),
			"BURST_SIZE should be <= %u", MAX_BURST);

	rte_bbdev_info_get(tp->dev_id, &info);

	TEST_ASSERT_SUCCESS((num_ops > info.drv.queue_size_lim),
			"NUM_OPS cannot exceed %u for this device",
			info.drv.queue_size_lim);

	bufs = &tp->op_params->q_bufs[GET_SOCKET(info.socket_id)][queue_id];
	
	//&op_params->q_bufs[socket_id][queue_id].inputs
//printf("bufs len %d\n",bufs->input.data->data_len);
	while (rte_atomic16_read(&tp->op_params->sync) == SYNC_WAIT)
		rte_pause();

	ret = rte_bbdev_dec_op_alloc_bulk(tp->op_params->mp, ops_enq, num_ops);
	TEST_ASSERT_SUCCESS(ret, "Allocation failed for %d ops", num_ops);

	/* For throughput tests we need to disable early termination */
	if (check_bit(ref_op->ldpc_dec.op_flags,
			RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE))
		ref_op->ldpc_dec.op_flags -=
				RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
	ref_op->ldpc_dec.iter_max = get_iter_max();
	ref_op->ldpc_dec.iter_count = ref_op->ldpc_dec.iter_max;

	//if (test_vector.op_type != RTE_BBDEV_OP_NONE)
		copy_reference_ldpc_dec_op(ops_enq, num_ops, 0, bufs->inputs,
				bufs->hard_outputs, bufs->soft_outputs,
				bufs->harq_inputs, bufs->harq_outputs, ref_op, p_offloadParams);

	/* Set counter to validate the ordering */
	for (j = 0; j < num_ops; ++j)
		ops_enq[j]->opaque_data = (void *)(uintptr_t)j;

	for (i = 0; i < TEST_REPETITIONS; ++i) {
		for (j = 0; j < num_ops; ++j) {
			if (!loopback)
				mbuf_reset(
				ops_enq[j]->ldpc_dec.hard_output.data);
			if (hc_out || loopback)
				mbuf_reset(
				ops_enq[j]->ldpc_dec.harq_combined_output.data);
		}
		if (extDdr) {
			bool preload = i == (TEST_REPETITIONS - 1);
			preload_harq_ddr(tp->dev_id, queue_id, ops_enq,
					num_ops, preload);
		}
		start_time = rte_rdtsc_precise();

		for (enq = 0, deq = 0; enq < num_ops;) {
			num_to_enq = burst_sz;

			if (unlikely(num_ops - enq < num_to_enq))
				num_to_enq = num_ops - enq;
				
				//printf("pmd lcore ldpc dec data %x\n", *ops_enq[enq]->ldpc_dec.input.addr);

			enq += rte_bbdev_enqueue_ldpc_dec_ops(tp->dev_id,
					queue_id, &ops_enq[enq], num_to_enq);

			deq += rte_bbdev_dequeue_ldpc_dec_ops(tp->dev_id,
					queue_id, &ops_deq[deq], enq - deq);

/*			ops_td = &ops_deq[enq]->ldpc_dec;
                        hard_output = &ops_td->hard_output;
 struct rte_mbuf *m = hard_output->data;	
	 printf("deq nb segs %d\n", m->nb_segs);
*/ 
                }

		/* dequeue the remaining */
		while (deq < enq) {
			deq += rte_bbdev_dequeue_ldpc_dec_ops(tp->dev_id,
					queue_id, &ops_deq[deq], enq - deq);
		}

		total_time += rte_rdtsc_precise() - start_time;
	}

	tp->iter_count = 0;
	/* get the max of iter_count for all dequeued ops */
	for (i = 0; i < num_ops; ++i) {
		tp->iter_count = RTE_MAX(ops_enq[i]->ldpc_dec.iter_count,
				tp->iter_count);
	}
	if (extDdr) {
		/* Read loopback is not thread safe */
		retrieve_harq_ddr(tp->dev_id, queue_id, ops_enq, num_ops);
	}

	//if (test_vector.op_type != RTE_BBDEV_OP_NONE) {
	ret = validate_ldpc_dec_op(ops_deq, num_ops, ref_op,
				tp->op_params->vector_mask, p_out);
		TEST_ASSERT_SUCCESS(ret, "Validation failed!");
	//}

	rte_bbdev_dec_op_free_bulk(ops_enq, num_ops);

	double tb_len_bits = calc_ldpc_dec_TB_size(ref_op);

	tp->ops_per_sec = ((double)num_ops * TEST_REPETITIONS) /
			((double)total_time / (double)rte_get_tsc_hz());
	tp->mbps = (((double)(num_ops * TEST_REPETITIONS * tb_len_bits)) /
			1000000.0) / ((double)total_time /
			(double)rte_get_tsc_hz());

	return TEST_SUCCESS;
}


static void
print_enc_throughput(struct thread_params *t_params, unsigned int used_cores)
{
	unsigned int iter = 0;
	double total_mops = 0, total_mbps = 0;

	for (iter = 0; iter < used_cores; iter++) {
		printf(
			"Throughput for core (%u): %.8lg Ops/s, %.8lg Mbps\n",
			t_params[iter].lcore_id, t_params[iter].ops_per_sec,
			t_params[iter].mbps);
		total_mops += t_params[iter].ops_per_sec;
		total_mbps += t_params[iter].mbps;
	}
	printf(
		"\nTotal throughput for %u cores: %.8lg MOPS, %.8lg Mbps\n",
		used_cores, total_mops, total_mbps);
}

/* Aggregate the performance results over the number of cores used */
static void
print_dec_throughput(struct thread_params *t_params, unsigned int used_cores)
{
	unsigned int core_idx = 0;
	double total_mops = 0, total_mbps = 0;
	uint8_t iter_count = 0;

	for (core_idx = 0; core_idx < used_cores; core_idx++) {
		printf(
			"Throughput for core (%u): %.8lg Ops/s, %.8lg Mbps @ max %u iterations\n",
			t_params[core_idx].lcore_id,
			t_params[core_idx].ops_per_sec,
			t_params[core_idx].mbps,
			t_params[core_idx].iter_count);
		total_mops += t_params[core_idx].ops_per_sec;
		total_mbps += t_params[core_idx].mbps;
		iter_count = RTE_MAX(iter_count,
				t_params[core_idx].iter_count);
	}
	printf(
		"\nTotal throughput for %u cores: %.8lg MOPS, %.8lg Mbps @ max %u iterations\n",
		used_cores, total_mops, total_mbps, iter_count);
}


/*
 * Test function that determines how long an enqueue + dequeue of a burst
 * takes on available lcores.
 */
int
start_pmd_dec(struct active_device *ad,
		struct test_op_params *op_params, 
                t_nrLDPCoffload_params *p_offloadParams,
		int8_t* p_out)
{
	int ret;
	unsigned int lcore_id, used_cores = 0;
	struct thread_params *t_params, *tp;
	struct rte_bbdev_info info;
	uint16_t num_lcores;
	const char *op_type_str;

	rte_bbdev_info_get(ad->dev_id, &info);

	op_type_str = rte_bbdev_op_type_str(test_vector.op_type);
	TEST_ASSERT_NOT_NULL(op_type_str, "Invalid op type: %u",
			test_vector.op_type);

	printf("+ ------------------------------------------------------- +\n");
	printf("== start pmd dec\ndev: %s, nb_queues: %u, burst size: %u, num ops: %u, num_lcores: %u, op type: %s, itr mode: %s, GHz: %lg\n",
			info.dev_name, ad->nb_queues, op_params->burst_sz,
			op_params->num_to_process, op_params->num_lcores,
			op_type_str,
			intr_enabled ? "Interrupt mode" : "PMD mode",
			(double)rte_get_tsc_hz() / 1000000000.0);

	/* Set number of lcores */
	num_lcores = (ad->nb_queues < (op_params->num_lcores))
			? ad->nb_queues
			: op_params->num_lcores;

	/* Allocate memory for thread parameters structure */
	t_params = rte_zmalloc(NULL, num_lcores * sizeof(struct thread_params),
			RTE_CACHE_LINE_SIZE);
	TEST_ASSERT_NOT_NULL(t_params, "Failed to alloc %zuB for t_params",
			RTE_ALIGN(sizeof(struct thread_params) * num_lcores,
				RTE_CACHE_LINE_SIZE));

	rte_atomic16_set(&op_params->sync, SYNC_WAIT);

	/* Master core is set at first entry */
	t_params[0].dev_id = ad->dev_id;
	t_params[0].lcore_id = rte_lcore_id();
	t_params[0].op_params = op_params;
	t_params[0].queue_id = ad->queue_ids[used_cores++];
	t_params[0].iter_count = 0;

	RTE_LCORE_FOREACH_SLAVE(lcore_id) {
		if (used_cores >= num_lcores)
			break;

		t_params[used_cores].dev_id = ad->dev_id;
		t_params[used_cores].lcore_id = lcore_id;
		t_params[used_cores].op_params = op_params;
		t_params[used_cores].queue_id = ad->queue_ids[used_cores];
		t_params[used_cores].iter_count = 0;

		rte_eal_remote_launch(pmd_lcore_ldpc_dec,
				&t_params[used_cores++], lcore_id);
	}

	rte_atomic16_set(&op_params->sync, SYNC_START);
	ret = pmd_lcore_ldpc_dec(&t_params[0], p_offloadParams, p_out);

	/* Master core is always used */
	for (used_cores = 1; used_cores < num_lcores; used_cores++)
		ret |= rte_eal_wait_lcore(t_params[used_cores].lcore_id);

	/* Return if test failed */
	if (ret) {
		rte_free(t_params);
		return ret;
	}

	/* Print throughput if interrupts are disabled and test passed */
	if (!intr_enabled) {
		if (test_vector.op_type == RTE_BBDEV_OP_TURBO_DEC ||
				test_vector.op_type == RTE_BBDEV_OP_LDPC_DEC)
			print_dec_throughput(t_params, num_lcores);
		else
			print_enc_throughput(t_params, num_lcores);
		rte_free(t_params);
		return ret;
	}

	/* In interrupt TC we need to wait for the interrupt callback to deqeue
	 * all pending operations. Skip waiting for queues which reported an
	 * error using processing_status variable.
	 * Wait for master lcore operations.
	 */
	tp = &t_params[0];
	while ((rte_atomic16_read(&tp->nb_dequeued) <
			op_params->num_to_process) &&
			(rte_atomic16_read(&tp->processing_status) !=
			TEST_FAILED))
		rte_pause();

	tp->ops_per_sec /= TEST_REPETITIONS;
	tp->mbps /= TEST_REPETITIONS;
	ret |= (int)rte_atomic16_read(&tp->processing_status);

	/* Wait for slave lcores operations */
	for (used_cores = 1; used_cores < num_lcores; used_cores++) {
		tp = &t_params[used_cores];

		while ((rte_atomic16_read(&tp->nb_dequeued) <
				op_params->num_to_process) &&
				(rte_atomic16_read(&tp->processing_status) !=
				TEST_FAILED))
			rte_pause();

		tp->ops_per_sec /= TEST_REPETITIONS;
		tp->mbps /= TEST_REPETITIONS;
		ret |= (int)rte_atomic16_read(&tp->processing_status);
	}

	/* Print throughput if test passed */
	if (!ret) {
		if (test_vector.op_type == RTE_BBDEV_OP_TURBO_DEC ||
				test_vector.op_type == RTE_BBDEV_OP_LDPC_DEC)
			print_dec_throughput(t_params, num_lcores);
		else if (test_vector.op_type == RTE_BBDEV_OP_TURBO_ENC ||
				test_vector.op_type == RTE_BBDEV_OP_LDPC_ENC)
			print_enc_throughput(t_params, num_lcores);
	}

	rte_free(t_params);
	return ret;
}


/* Declare structure for command line test parameters and options */
static struct test_params {
	struct test_command *test_to_run[8];
	unsigned int num_tests;
	unsigned int num_ops;
	unsigned int burst_sz;
	unsigned int num_lcores;
	double snr;
	unsigned int iter_max;
	char test_vector_filename[PATH_MAX];
	bool init_device;
} test_params;

const char *
get_vector_filename(void)
{
	return test_params.test_vector_filename;
}
unsigned int
get_num_ops(void)
{
	return test_params.num_ops;
}

unsigned int
get_burst_sz(void)
{
	return test_params.burst_sz;
}

unsigned int
get_num_lcores(void)
{
	return test_params.num_lcores;
}

double
get_snr(void)
{
	return test_params.snr;
}

unsigned int
get_iter_max(void)
{
	return test_params.iter_max;
}

bool
get_init_device(void)
{
	return test_params.init_device;
}

int count_init = 0;

int32_t nrLDPC_decod_offload(t_nrLDPC_dec_params* p_decParams, uint8_t C, uint8_t rv, uint16_t F, uint32_t E, uint8_t Qm, int8_t* p_llr, int8_t* p_out)
{
    	t_nrLDPCoffload_params offloadParams;
    	t_nrLDPCoffload_params* p_offloadParams    = &offloadParams;

    	uint32_t numIter = 0;
	int ret;

int argc_re=4;
char *argv_re[15];
argv_re[0] = "/home/wang/dpdk2005/dpdk-20.05/build/app/testbbdev";
argv_re[1] = "--"; //./build/app/testbbdev"; 
//argv_re[2] = "-c";
//argv_re[3] = "throughput";
argv_re[2] = "-v";
argv_re[3] = "../../../targets/ARCH/test-bbdev/test_vectors/ldpc_dec_v8480.data";
if (count_init == 0) {

printf("argcre %d argvre %s %s %s %s\n", argc_re, argv_re[0], argv_re[1], argv_re[2], argv_re[3]);
count_init++;
        ret = rte_eal_init(argc_re, argv_re);
}
argc_re = 3;
argv_re[0] = "--";  
argv_re[1] = "-v";
argv_re[2] = "../../../targets/ARCH/test-bbdev/test_vectors/ldpc_dec_v8480.data";


	memset(&test_vector_dec, 0, sizeof(struct test_bbdev_vector));

struct rte_bbdev_op_ldpc_dec *ldpc_dec = &test_vector_dec.ldpc_dec;
test_vector.op_type = RTE_BBDEV_OP_LDPC_DEC; //vector->op_type;
test_vector.expected_status = 0;

 ldpc_dec->cb_params.e = E; 
ldpc_dec->iter_count = 3;
                ldpc_dec->basegraph = p_decParams->BG; 
                ldpc_dec->z_c = p_decParams->Z; 
                ldpc_dec->q_m = Qm;
                ldpc_dec->n_filler = F; 
                ldpc_dec->n_cb = (p_decParams->BG==1)?(66*p_decParams->Z):(50*p_decParams->Z); 
                ldpc_dec->iter_max = 8; 
                ldpc_dec->rv_index = rv; 
                ldpc_dec->op_flags = RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
                ldpc_dec->code_block_mode = (C>1)?1:0; 
ldpc_dec->tb_params.ea = E;
ldpc_dec->tb_params.eb = E;
ldpc_dec->tb_params.r = 0;


	test_params.num_ops=1; 
	test_params.burst_sz=1;
	test_params.num_lcores=1;		
	test_params.num_tests = 1;

	device_setup();
	ut_setup();

	p_offloadParams->E = E;
	p_offloadParams->n_cb = (p_decParams->BG==1)?(66*p_decParams->Z):(50*p_decParams->Z);
	p_offloadParams->BG = p_decParams->BG;
	p_offloadParams->Z = p_decParams->Z;
	p_offloadParams->rv = rv;
	p_offloadParams->F = F;
	p_offloadParams->Qm = Qm;
	ldpc_decod_ut(p_llr, p_offloadParams, p_out);

	//ut_teardown();

    return numIter;
}
