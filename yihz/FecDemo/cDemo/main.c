#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <hugetlbfs.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/time.h>

#include "logger_wrapper.h"
#include "ff_define.h"
#include "mem.h"
#include "dataFile.h"
#include "fec_c_if.h"



int32_t zlogMask=0xF00000; //driver log all open


char FecDeDataFileName[]="decode_data_0.bin";
char FecEnDataFileName[]="encode_data_0.bin";
char saveDataFileName[]="save_data.bin";



//
//
//
//
////////////////////////////////////////////////////////
void dm_time_rec_add(struct timespec *t)
{
    clock_gettime(CLOCK_MONOTONIC, t); 
}

uint64_t dm_time_rec_diff(struct timespec* now, struct timespec* old) {
    uint64_t diff_time = 0;
    diff_time = MILLION * (now->tv_sec - old->tv_sec) + (now->tv_nsec - old->tv_nsec) / THOUSAND;
    return diff_time;
}

volatile TimeStempIndUnion gTxTimeIndex;
volatile uint64_t *pTxTimeIndex = NULL;
volatile int32_t newSlotIdx,oldSlotIdx;
volatile uint32_t tti_loop_count=0;

volatile uint64_t loopCountEn=0,loopCountDe=0;
volatile uint64_t DeAllTime=0, EnAllTime=0;
volatile uint64_t DeAllLen=0, EnAllLen=0;

int32_t lstSlotIdx =0;


#define DE_RESULT_DATA_MAX 0x100000
#define EN_RESULT_DATA_MAX 0x10000
unsigned char DeDataResult[DE_RESULT_DATA_MAX];
unsigned char EnDataResult[EN_RESULT_DATA_MAX];
char DeResultFileName[]="decode_00.bin";
char EnResultFileName[]="encode_00.bin";

//
//
//
//
////////////////////////////////////////////////////////
int main(int argc,char *argv[])
{
    int32_t retVal,i;
    uint64_t DmaBaseVirAddr,DmaBasePhyAddr, TtiPhyAddr;
    uint8_t *pTtiVirAddr;
    MemorySegmentStruct *pFecSegentInfo;
    HugepageTblStruct *pFecHugepageTbl;
    DescriptorAddrRingStruct DescRingAddr={0};
    uint64_t time_dis = 0;
    uint32_t fpgaData1,fpgaData2;
    DescDataInfo EnDataTx, EnDataRx;
    DescDataInfo DeDataTx, DeDataRx;
    DataHeadInfo *pDataHeadInfo;


    struct timespec deRxEndNew;
    struct timespec deRxEndOld;
    struct timespec deTxEndNew;
    struct timespec deTxEndOld;
    struct timespec enRxEndNew;
    struct timespec enRxEndOld;
    struct timespec enTxEndNew;
    struct timespec enTxEndOld;

    struct timespec deTxStartRec;
    struct timespec deTxStartNew;
    struct timespec deRxStartRec;
    struct timespec deRxStartNew;
    struct timespec enTxStartRec;
    struct timespec enTxStartNew;
    struct timespec enRxStartRec;
    struct timespec enRxStartNew;

    volatile uint32_t k;


	

    mm_huge_table_init(&pFecHugepageTbl, SW_FPGA_FH_TOTAL_BUFFER_LEN);
    printf( "%s>>>---pFecHugepageTbl->nPagePhyAddr=0x%lx\n",__FUNCTION__,(uint64_t)pFecHugepageTbl->nPagePhyAddr );
    mm_segment_init(&pFecSegentInfo, pFecHugepageTbl);
    printf("%s>>> pFecSegentInfo->pSegment=0x%lx, pFecSegentInfo->nPhysicalAddr=0x%lx\n", __FUNCTION__, pFecSegentInfo->pSegment, pFecSegentInfo->nPhysicalAddr);
    mm_regist_addr_to_ring(&DescRingAddr, pFecSegentInfo);

    //log init
    GLOBAL_FRLOG_INIT("/tmp/usrp/l1/", "l1", 2, 3);
    //
    dev2_init_fec();
    //
    dev2_register_mem_addr(&DescRingAddr);
    // malloc memory for rx/tx data
    get_data_addr_len(&DescRingAddr, &EnDataTx, &EnDataRx, &DeDataTx, &DeDataRx);

    // date init
    dm_time_rec_add(&enTxStartNew);
    dm_time_rec_add(&enRxStartNew);
    dm_time_rec_add(&deTxEndNew);
    dm_time_rec_add(&deRxEndNew);

    // test data load
    getBinDataFromFile((unsigned char *)DeDataTx.nVirtAddr, &DeDataTx.dataLen, FecDeDataFileName);
    getBinDataFromFile((unsigned char *)EnDataTx.nVirtAddr, &EnDataTx.dataLen, FecEnDataFileName);
    //load_expect_data(DeDataResult,DeResultFileName);
    //load_expect_data(EnDataResult,EnResultFileName);

    pDataHeadInfo=(DataHeadInfo *)DeDataRx.nVirtAddr;
    // main loop !!!
    while(1)
    {
        // Fec Encode Ring
        if( fec_en_tx_release() )
        {
            // release encode data buffer
        }
        if( fec_en_rx_release() )
        {
            if( zlogMask & ZLOG_DRIVER_DM_FEC ){
                dm_time_rec_add(&enRxEndNew);
                time_dis = dm_time_rec_diff(&enRxEndNew, &enTxStartNew);
                EnAllTime+=time_dis;
                EnAllLen+=EnDataTx.dataLen;
                loopCountEn++;
                if(loopCountEn>=10000){
                     EnAllTime/=1000;
                     zLog(PHY_LOG_INFO,"[FecEnRxRing] data throughput=%dMbps current time=%dus EnDataTx.dataLen=%d ---[DRV_LOG]", 8*EnAllLen/EnAllTime/1024,time_dis,EnDataTx.dataLen);
                     //zLog(PHY_LOG_INFO,"[FecEnRxRing] EnAllTime=%d EnAllLen=%d ---[DRV_LOG]", EnAllTime, EnAllLen);
                     loopCountEn=0;
                     EnAllTime=0;
                     EnAllLen=0;
                }
            }
            saveBinData((unsigned char *)EnDataRx.nVirtAddr, FEC_ENCODE_DATA);
        }
        // Fec Decode Ring
        if( fec_de_tx_release() )
        {
            // release decode data buffer
        }
        if( fec_de_rx_release() )
        {
            if( zlogMask & ZLOG_DRIVER_DM_FEC ){
                dm_time_rec_add(&deRxEndNew);
                time_dis = dm_time_rec_diff(&deRxEndNew, &deTxStartNew);
                DeAllTime+=time_dis;
                DeAllLen+=pDataHeadInfo->pktLen;
                loopCountDe++;
                if(loopCountDe>=10000){
                     DeAllTime/=1000;
                     zLog(PHY_LOG_INFO,"[FecDeRxRing] data throughput=%dMbps ,current time=%dus,pDataHeadInfo->pktLen=%d ---[DRV_LOG]", 8*DeAllLen/DeAllTime/1024, time_dis, pDataHeadInfo->pktLen);
                     //zLog(PHY_LOG_INFO,"[FecDeRxRing] DeAllTime=%d DeAllLen=%d ----[DRV_LOG]", DeAllTime, DeAllLen);
                     loopCountDe=0;
                     DeAllTime=0;
                     DeAllLen=0;
                }
            }
            //compare_data();
            saveBinData((unsigned char *)DeDataRx.nVirtAddr, FEC_DECODE_DATA);
        }

        //FEC Encode
        if( fec_en_tx_require() ) 
        {
            if( zlogMask & ZLOG_DRIVER_DM_FEC ){
                 dm_time_rec_add(&enTxStartNew);
            }
            dev2_send_en_data(&EnDataTx);
        }
        if( fec_en_rx_require() )
        {
            dev2_recv_en_data(&EnDataRx);
        }

        // FEC Decode
        if( fec_de_tx_require() )
        {
            if( zlogMask & ZLOG_DRIVER_DM_FEC ){
                dm_time_rec_add(&deTxStartNew);
            }
            dev2_send_de_data(&DeDataTx);
        }
        if( fec_de_rx_require() )
        {
            dev2_recv_de_data(&DeDataRx);
        }
    }
    return 0;
}


