#include <cstdio>
#include <ctime>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdint>

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
#include "nr_sys_info.h"
#include "nr_rf_chan.h"
#include "fec_c_if.h"

using namespace std;



int32_t zlogMask=0xF00000; //driver log all open


char FecDeDataFileName[]="decode_data_3.bin";
char FecEnDataFileName[]="encode_data_2.bin";
//char fecDataFileName[]="rx_decode_In_slot_Counter0312.bin";
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

int32_t lstSlotIdx =0;


//
//
//
//
////////////////////////////////////////////////////////
int get_tti_valid()
{
    int32_t slot_step,ret;

    ret=0;
    gTxTimeIndex.nAll = *pTxTimeIndex;
    newSlotIdx=(gTxTimeIndex.bit.nFrameIdx*10 + gTxTimeIndex.bit.nSubFrameIdx) * 2 + gTxTimeIndex.bit.nSlotIdx;
    if(lstSlotIdx==0){
        oldSlotIdx=newSlotIdx;
        lstSlotIdx=1;
        return 3;
    }
    slot_step=newSlotIdx-oldSlotIdx;
    tti_loop_count++;
    if(slot_step>1)
    {
        zLog(PHY_LOG_ERROR,"[TTI] slot_step=%d, tti_loop_count=0x%x, nFrameIdx=%d,nSubFrameIdx=%d,nSlotIdx=%d,nSymbolIdx=%d -------[DRV_LOG]",slot_step, tti_loop_count, gTxTimeIndex.bit.nFrameIdx,gTxTimeIndex.bit.nSubFrameIdx,gTxTimeIndex.bit.nSlotIdx,gTxTimeIndex.bit.nSymbolIdx);
        ret=2;
    }else if(slot_step==1){
       zLog(PHY_LOG_INFO,"[TTI] slot_step=%d, tti_loop_count=0x%x, nFrameIdx=%d,nSubFrameIdx=%d,nSlotIdx=%d,nSymbolIdx=%d -------[DRV_LOG]",slot_step, tti_loop_count, gTxTimeIndex.bit.nFrameIdx,gTxTimeIndex.bit.nSubFrameIdx,gTxTimeIndex.bit.nSlotIdx,gTxTimeIndex.bit.nSymbolIdx);
       ret=1;
    }else{
       //zLog(PHY_LOG_INFO,"[TTI] slot_step=%d, tti_loop_count=%d, nFrameIdx=%d,nSubFrameIdx=%d,nSlotIdx=%d,nSymbolIdx=%d -------[DRV_LOG]",slot_step, tti_loop_count, gTxTimeIndex.bit.nFrameIdx,gTxTimeIndex.bit.nSubFrameIdx,gTxTimeIndex.bit.nSlotIdx,gTxTimeIndex.bit.nSymbolIdx);
       ret=0;
    }
    oldSlotIdx=newSlotIdx;
    return ret;
}

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
    MemorySegmentStruct *pFecSegentInfo,*pFthSegentInfo;
    HugepageTblStruct *pFecHugepageTbl,*pFthHugepageTbl;
    DescriptorAddrRingStruct DescRingAddr={0};
    uint64_t time_dis = 0;
    uint32_t fpgaData1,fpgaData2;
    DescDataInfo EnDataTx, EnDataRx;
    DescDataInfo DeDataTx, DeDataRx;

    volatile uint32_t nStatFecTxEnDma = IS_FREE;
    volatile uint32_t nStatFecRxEnDma = IS_FREE;
    volatile uint32_t nStatFecTxDeDma = IS_FREE;
    volatile uint32_t nStatFecRxDeDma = IS_FREE;
    volatile uint32_t nStatFthTxDma = IS_FREE;
    volatile uint32_t nStatFthRxDma = IS_FREE;
    DescChainInfo *pFecTxEnRing = NULL;
    DescChainInfo *pFecRxEnRing = NULL;
    DescChainInfo *pFecTxDeRing = NULL;
    DescChainInfo *pFecRxDeRing = NULL;
    DescChainInfo *pFthTxRing = NULL;
    DescChainInfo *pFthRxRing = NULL;

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

    uint8_t tmpData;
    volatile uint32_t k,m;
    volatile int count_en=0;
    volatile int count_de=0;


	

    mm_huge_table_init(&pFecHugepageTbl, SW_FPGA_FH_TOTAL_BUFFER_LEN);
    mm_huge_table_init(&pFthHugepageTbl, SW_FPGA_FH_TOTAL_BUFFER_LEN);
    printf("%s>>>---pFecHugepageTbl->nPagePhyAddr=0x%lx pFthHugepageTbl->nPagePhyAddr=0x%lx---\n",__FUNCTION__,(uint64_t)pFecHugepageTbl->nPagePhyAddr,(uint64_t)pFthHugepageTbl->nPagePhyAddr);
    //printf(">>>huge init ok.\n");
    mm_segment_init(&pFecSegentInfo, pFecHugepageTbl);
    mm_segment_init(&pFthSegentInfo, pFthHugepageTbl);
    printf("%s>>> pFecSegentInfo->pSegment=0x%lx, pFecSegentInfo->nPhysicalAddr=0x%lx\n", __FUNCTION__, pFecSegentInfo->pSegment, pFecSegentInfo->nPhysicalAddr);
    printf("%s>>> pFthSegentInfo->pSegment=0x%lx, pFthSegentInfo->nPhysicalAddr=0x%lx\n", __FUNCTION__, pFthSegentInfo->pSegment, pFthSegentInfo->nPhysicalAddr);
    //printf(">>>segment init ok.\n");
    mm_regist_addr_to_ring(&DescRingAddr,pFthSegentInfo,pFecSegentInfo);
    printf(">>>DescRingAddr.TxEnCodeRingBuffer.pPageVirtAddr=0x%lx\n",DescRingAddr.TxEnCodeRingBuffer.pPageVirtAddr);
    get_data_addr_len(&DescRingAddr, &EnDataTx, &EnDataRx, &DeDataTx, &DeDataRx);


    //log init
    GLOBAL_FRLOG_INIT("/tmp/usrp/l1/", "l1", 2, 3);
    //
    dev2_init_fec();
    //
    dev2_register_ring_addr(&DescRingAddr);
	



    // test date load
    dm_time_rec_add(&enTxStartNew);
    dm_time_rec_add(&enRxStartNew);
    dm_time_rec_add(&deTxEndNew);
    dm_time_rec_add(&deRxEndNew);

    getBinDataFromFile((unsigned char *)DeDataTx.nVirtAddr, &DeDataTx.dataLen, FecDeDataFileName);
    getBinDataFromFile((unsigned char *)EnDataTx.nVirtAddr, &EnDataTx.dataLen, FecEnDataFileName);
    load_expect_data(DeDataResult,DeResultFileName);
    load_expect_data(EnDataResult,EnResultFileName);

    // main loop !!!
    while(1)
    {
        //tti process. return 0:tti step=0, 1:tti step=1, 2:tti step>1
        retVal=get_tti_valid();
        if(retVal!=1) continue;
        //for(k=0;k<0x40000;k++); //0x4000==25us

        // Fec Encode Ring
        if( fec_en_tx_release() )
        {
            if(zlogMask & ZLOG_DRIVER_DM_FEC){
                enTxEndOld.tv_sec =enTxEndNew.tv_sec;
                enTxEndOld.tv_nsec=enTxEndNew.tv_nsec;
                dm_time_rec_add(&enTxEndNew);
                time_dis = dm_time_rec_diff(&enTxEndNew, &enTxEndOld);
                zLog(PHY_LOG_INFO,"[FecEnTxRing] time=%d ---[DRV_LOG]",time_dis);
            }
        }
        if( fec_en_rx_release() )
        {
            if( zlogMask & ZLOG_DRIVER_DM_FEC ){
                enRxEndOld.tv_sec =enRxEndNew.tv_sec;
                enRxEndOld.tv_nsec=enRxEndNew.tv_nsec;
                dm_time_rec_add(&enRxEndNew);
                time_dis = dm_time_rec_diff(&enRxEndNew, &enRxEndOld);
                zLog(PHY_LOG_INFO,"[FecEnRxRing] time=%d ---[DRV_LOG]", time_dis);
            }
            saveBinData((unsigned char *)EnDataRx.nVirtAddr, FEC_ENCODE_DATA);
        }
        // Fec Decode Ring
        if( fec_de_tx_release() )
        {
            if( zlogMask & ZLOG_DRIVER_DM_FEC ){
                deTxEndOld.tv_sec =deTxEndNew.tv_sec;
                deTxEndOld.tv_nsec=deTxEndNew.tv_nsec;
                dm_time_rec_add(&deTxEndNew);
                time_dis = dm_time_rec_diff(&deTxEndNew, &deTxEndOld);
                zLog(PHY_LOG_INFO,"[FecDeTxRing] time=%d -----[DRV_LOG]",time_dis);
            }
        }
        if( fec_de_rx_release() )
        {
            if( zlogMask & ZLOG_DRIVER_DM_FEC ){
                deRxEndOld.tv_sec =deRxEndNew.tv_sec;
                deRxEndOld.tv_nsec=deRxEndNew.tv_nsec;
                dm_time_rec_add(&deRxEndNew);
                time_dis = dm_time_rec_diff(&deRxEndNew, &deRxEndOld);
                zLog(PHY_LOG_INFO,"[FecDeRxRing] time=%d ----[DRV_LOG]", time_dis);
            }
            //compare_data();
            saveBinData((unsigned char *)DeDataRx.nVirtAddr, FEC_DECODE_DATA);
        }

        //FEC Encode
        if( fec_en_tx_require() ) 
        {
            if( zlogMask & ZLOG_DRIVER_DM_FEC ){
                enTxStartRec.tv_sec =enTxStartNew.tv_sec;
                enTxStartRec.tv_nsec=enTxStartNew.tv_nsec;
                 dm_time_rec_add(&enTxStartNew);
                 time_dis = dm_time_rec_diff(&enTxStartNew, &enTxStartRec);
                 zLog(PHY_LOG_INFO,"[FecEnTX] time=%d -----------[DRV_LOG]",time_dis);
            }
            dev2_send_en_data(&EnDataTx);
        }
        if( fec_en_rx_require() )
        {
            if( zlogMask & ZLOG_DRIVER_DM_FEC ){
                enRxStartRec.tv_sec =enRxStartNew.tv_sec;
                enRxStartRec.tv_nsec=enRxStartNew.tv_nsec;
                dm_time_rec_add(&enRxStartNew);
                time_dis = dm_time_rec_diff(&enRxStartNew, &enRxStartRec);
                zLog(PHY_LOG_INFO,"[FecEnRX] time=%d -----------[DRV_LOG]",time_dis);
            }
            dev2_recv_en_data(&EnDataRx);
        }

        // FEC Decode
        if( fec_de_tx_require() )
        {
            if( zlogMask & ZLOG_DRIVER_DM_FEC ){
                deTxStartRec.tv_sec =deTxStartNew.tv_sec;
                deTxStartRec.tv_nsec=deTxStartNew.tv_nsec;
                dm_time_rec_add(&deTxStartNew);
                time_dis = dm_time_rec_diff(&deTxStartNew, &deTxStartRec);
                zLog(PHY_LOG_INFO,"[FecDeTx] time=%d  ----[DRV_LOG]",time_dis);
            }
            dev2_send_de_data(&DeDataRx);
        }
        if( fec_de_rx_require() )
        {
            if( zlogMask & ZLOG_DRIVER_DM_FEC ){
                deRxStartRec.tv_sec =deRxStartNew.tv_sec;
                deRxStartRec.tv_nsec=deRxStartNew.tv_nsec;
                dm_time_rec_add(&deRxStartNew);
                time_dis = dm_time_rec_diff(&deRxStartNew, &deRxStartRec);
                zLog(PHY_LOG_INFO,"[FecDeRx] time=%d -----------[DRV_LOG]",time_dis);
            }
            dev2_recv_de_data(&DeDataRx);
        }
    }
    return 0;
}


