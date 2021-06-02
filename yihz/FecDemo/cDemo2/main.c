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

//#include "logger_wrapper.h"
#include <dev2.0/logger_wrapper.h>
#include "ff_define.h"
#include "mem.h"
#include "dataFile.h"
//#include "fec_c_if.h"
#include <dev2.0/fec_c_if.h>



int32_t zlogMask=0xF00000; //driver log all open
static int  demo_count = 0;

// char FecDeDataFileName[]="encode_data_0.bin";
char FecEnDataFileName[]="oai_encode_data_0.bin";
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


volatile uint64_t loopCountEn=0,loopCountDe=0;
volatile uint64_t DeAllTime=0, EnAllTime=0;
volatile uint64_t DeAllLen=0, EnAllLen=0;

int32_t lstSlotIdx =0;

#define DEFAULT_EN_DATA_LEN  0x400000
#define DEFAULT_DE_DATA_LEN  0x400000
char DeResultFileName[]="decode_00.bin";
char EnResultFileName[]="encode_00.bin";
char HeadDataFileName[]="head_data.bin";
DescDataInfo EnDataTx,EnDataRx;
DescDataInfo DeDataTx,DeDataRx;

int step1=0,step2=0,step3=0,step4=0;
//
//
//
//
////////////////////////////////////////////////////////
int encoder_load( EncodeInHeaderStruct *pHeader, unsigned char * pSrc, unsigned char * pDst )
{
    EncodeInHeaderStruct *pDataHeadInfo;
    // if(demo_count > 1)
    // {
    //     return 1;
    // }
    

    //while(1)
    //{
        // Fec Encode Ring

      while(1){
        //FEC Encode
        if( fec_en_tx_require() ) 
        {
            memcpy( (unsigned char *)EnDataTx.nVirtAddr, pHeader, sizeof(EncodeInHeaderStruct));
            memcpy( (unsigned char *)EnDataTx.nVirtAddr+sizeof(EncodeInHeaderStruct), pSrc, pHeader->pktLen);
            EnDataTx.dataLen=pHeader->pktLen;
            dev2_send_en_data(&EnDataTx);
            //zLog(PHY_LOG_INFO," encoder_load:<1> ---[DRV_LOG]");
            break;
        }
      }
        

     while(1){
        if( fec_en_rx_require() )
        {
            //printf("fec_en_rx_require_start\n");
            dev2_recv_en_data(&EnDataRx);
            //zLog(PHY_LOG_INFO," encoder_load:<2> ---[DRV_LOG]");
            // printf("fec_en_rx_require_end\n");
            break;
        }
        //usleep(2);
      }

      while(1){
        if( fec_en_tx_release() )
        {
            // release encode data buffer
            //zLog(PHY_LOG_INFO," encoder_load:<3> ---[DRV_LOG]");
            // printf("fec_en_tx_release\n");
            break;
        }
      }


      while(1){
        if( fec_en_rx_release() )
        {
            //zLog(PHY_LOG_INFO," encoder_load:<3.5> ---[DRV_LOG]");
            // printf("fec_en_rx_release_start\n");
            pDataHeadInfo=(EncodeOutHeaderStruct *)EnDataRx.nVirtAddr;
            // memcpy(pDst, (unsigned char *)(EnDataRx.nVirtAddr), (pDataHeadInfo->pktLen));
            memcpy(pDst, (unsigned char *)(EnDataRx.nVirtAddr+32), (pDataHeadInfo->pktLen-32));
            //zLog(PHY_LOG_INFO," encoder_load:<4> ---[DRV_LOG]");
            // printf("fec_en_rx_release\n");
            return 1;
        }
      }
    //}
}

int decoder_load( DecodeInHeaderStruct *pHeader, unsigned char * pSrc, unsigned char * pDst )
{
    DecodeInHeaderStruct *pDataHeadInfo;
    // if(demo_count > 1)
    // {
    //     return 1;
    // }
    

    //while(1)
    //{
        // Fec Decode Ring

      while(1){
        //FEC Decode
        if( fec_de_tx_require() ) 
        {
            memcpy( (unsigned char *)DeDataTx.nVirtAddr, pHeader, sizeof(DecodeInHeaderStruct));
            memcpy( (unsigned char *)DeDataTx.nVirtAddr+sizeof(DecodeInHeaderStruct), pSrc, pHeader->pktLen);
            DeDataTx.dataLen=pHeader->pktLen;
            dev2_send_de_data(&DeDataTx);
            zLog(PHY_LOG_INFO," encoder_load:<1> ---[DRV_LOG]");
            break;
        }
      }
        

     while(1){
        if( fec_de_rx_require() )
        {
            //printf("fec_de_rx_require_start\n");
            dev2_recv_de_data(&DeDataRx);
            zLog(PHY_LOG_INFO," decoder_load:<2> ---[DRV_LOG]");
            // printf("fec_de_rx_require_end\n");
            break;
        }
        //usleep(2);
      }

      while(1){
        if( fec_de_tx_release() )
        {
            // release decode data buffer
            zLog(PHY_LOG_INFO," decoder_load:<3> ---[DRV_LOG]");
            // printf("fec_de_tx_release\n");
            break;
        }
      }


      while(1){
        if( fec_de_rx_release() )
        {
            //zLog(PHY_LOG_INFO," decoder_load:<3.5> ---[DRV_LOG]");
            // printf("fec_de_rx_release_start\n");
            pDataHeadInfo=(DecodeOutHeaderStruct *)DeDataRx.nVirtAddr;
            //memcpy(pDst, (unsigned char *)(DeDataRx.nVirtAddr), (pDataHeadInfo->pktLen));
            memcpy(pDst, (unsigned char *)(DeDataRx.nVirtAddr+32), (pDataHeadInfo->pktLen-32));
            zLog(PHY_LOG_INFO," decoder_load:<4> ---[DRV_LOG]");
            // printf("fec_de_rx_release\n");
            return 1;
        }
      }
    //}
}




void encode_tx_head( EncodeInHeaderStruct *pHD )
{
    //word 0
    pHD->pktType=0x12;
    pHD->rsv0=0x00;
    pHD->chkCode=0xFAFA;
    //word 1
    pHD->pktLen=0x1000;
    pHD->rsv1=0x0000;
    //word 2
    pHD->rsv2=0x0;
    pHD->sectorId=0x0;
    pHD->rsv3=0x0;
    //word 3
    pHD->sfn=0x13c;
    pHD->rsv4=0x0;
    pHD->subfn=0x1;
    pHD->slotNum=0x2;
    pHD->pduIdx=0x0;
    pHD->rev5=0x0;
    //word 4
    pHD->tbSizeB=0x0fc1;
    pHD->rev6=0x0;
    pHD->lastTb=0x1;
    pHD->firstTb=0x1;
    pHD->rev7=0x0;
    pHD->cbNum=0x04;
    //word 5
    pHD->qm=0x3;
    pHD->rev8=0x0;
    pHD->fillbit=0x160;
    pHD->rev9=0x0;
    pHD->kpInByte=0x3f4;
    pHD->rev10=0x0;
    //word 6
    pHD->gamma=0x02;
    pHD->codeRate=0x2e;
    pHD->rev11=0x0;
    pHD->rvIdx=0x0;
    pHD->rev12=0x0;
    pHD->lfSizeIx=0x7;
    pHD->rev13=0x0;
    pHD->iLs=0x1;
    pHD->bg=0x0;
    //word 7
    pHD->e1=0x44be;
    pHD->e0=0x44b8;

}

void decode_tx_head( DecodeInHeaderStruct *pHD )
{
    //word 0
    pHD->pktType=0x10;
    pHD->rsv0=0x00;
    pHD->chkCode=0xFAFA;
    //word 1
    pHD->pktLen=0x1000;
    pHD->rsv1=0x0000;
    //word 2
    pHD->pktTpTmp=0x0;
    pHD->pduSize=0x0;
    pHD->sectorId=0x0;
    pHD->rsv2=0x0;
    //word 3
    pHD->sfn=0x13c;
    pHD->rsv3=0x0;
    pHD->subfn=0x1;
    pHD->slotNum=0x2;
    pHD->pduIdx=0x0;
    pHD->rev4=0x0;
    //word 4
    pHD->tbSizeB=0x0fc1;
    pHD->rev5=0x0;
    pHD->lastTb=0x1;
    pHD->firstTb=0x1;
    pHD->rev6=0x0;
    pHD->cbNum=0x04;
    //word 5
    pHD->qm=0x3;
    pHD->rev7=0x0;
    pHD->fillbit=0x160;
    pHD->kpInByte=0x3f4;
    //word 6
    pHD->gamma=0x02;
    pHD->maxRowNm=0x2E;
    pHD->maxRvIdx=0x0;
    pHD->rvIdx=0x0;
    pHD->ndi=0x0;
    pHD->flush=0x0;
    pHD->maxIter=0x0;
    pHD->lfSizeIx=0x7;
    pHD->rev10=0x0;
    pHD->iLs=0x1;
    pHD->bg=0x0;
    //word 7
    pHD->e1=0x44be;
    pHD->e0=0x44b8;

}

//
//
//
//
////////////////////////////////////////////////////////
int main(int argc,char *argv[])
{
    volatile uint32_t k;
    int32_t retVal,i;
    MemorySegmentStruct *pFecSegentInfo;
    HugepageTblStruct *pFecHugepageTbl;
    DescriptorAddrRingStruct DescRingAddr={0};
    uint64_t  time_dis =0;

    struct timespec deEndTime,deStartTime;
    struct timespec enEndTime,enStartTime;
    EncodeInHeaderStruct EncodeHeadData;
    unsigned char *pEnDataSrc,*pEnDataDst;
    int dataLen=0;
	

    mm_huge_table_init(&pFecHugepageTbl, SW_FPGA_FH_TOTAL_BUFFER_LEN);
    printf( "%s>>>---pFecHugepageTbl->nPagePhyAddr=0x%lx\n",__FUNCTION__,(uint64_t)pFecHugepageTbl->nPagePhyAddr );
    mm_segment_init(&pFecSegentInfo, pFecHugepageTbl);
    printf("%s>>> pFecSegentInfo->pSegment=0x%lx, pFecSegentInfo->nPhysicalAddr=0x%lx\n", __FUNCTION__, pFecSegentInfo->pSegment, pFecSegentInfo->nPhysicalAddr);
    mm_regist_addr_to_ring(&DescRingAddr, pFecSegentInfo);

    //log init
    GLOBAL_FRLOG_INIT("/tmp/usrp/l1/", "l1", 2, 3);
    dev2_init_fec();
    dev2_register_mem_addr(&DescRingAddr);
    get_data_addr_len(&DescRingAddr, &EnDataTx, &EnDataRx, &DeDataTx, &DeDataRx);

    // malloc memory for rx/tx data
    pEnDataSrc=(unsigned char *)malloc(DEFAULT_EN_DATA_LEN);
    pEnDataDst=(unsigned char *)malloc(DEFAULT_EN_DATA_LEN);
    // test data load
    getBinDataFromFile( pEnDataSrc, &dataLen, FecEnDataFileName );
    //create head;
    encode_tx_head(&EncodeHeadData);

    //saveBinDataTofile(&EncodeHeadData, sizeof(EncodeInHeaderStruct), HeadDataFileName);
    //printf("head data file save.\n");
    //sleep(100);
    // main loop !!!
    //while( loopCountEn<3 )
    // while( 1 )
    // {
        // Fec Encode
        dm_time_rec_add(&enStartTime);
        encoder_load( &EncodeHeadData, pEnDataSrc, pEnDataDst );
        dm_time_rec_add(&enEndTime);
        time_dis = dm_time_rec_diff(&enEndTime, &enStartTime);

        EnAllTime+=time_dis;
        EnAllLen+=dataLen;
        loopCountEn++;
        if(loopCountEn>=10000)
        {
            EnAllTime/=1000;
            zLog(PHY_LOG_INFO,"[FecEnRxRing] data throughput=%dMbps current time=%dus dataLen=%d ---[DRV_LOG]", 8*EnAllLen/EnAllTime/1000,time_dis, dataLen);
            loopCountEn=0;
            EnAllTime=0;
            EnAllLen=0;
        }
        printf("---end---\n");
        // for(i=0; i<10;i++){
        //     printf("%d\n",i);
        // }
        saveBinData(pEnDataDst, FEC_ENCODE_DATA);
    // }
    // printf("---end---\n");
    // sleep(1);
    return 0;
}

int HugePage_Init(int bbb)
{
    // demo_count++;
    // if(demo_count > 1)
    // {
    //     return 0;
    // }
    volatile uint32_t k;
    int32_t retVal,i;
    MemorySegmentStruct *pFecSegentInfo;
    HugepageTblStruct *pFecHugepageTbl;
    DescriptorAddrRingStruct DescRingAddr={0};
    uint64_t  time_dis =0;

    struct timespec deEndTime,deStartTime;
    struct timespec enEndTime,enStartTime;
    EncodeInHeaderStruct EncodeHeadData;
    unsigned char *pEnDataSrc,*pEnDataDst;
    int dataLen=0;
	

    mm_huge_table_init(&pFecHugepageTbl, SW_FPGA_FH_TOTAL_BUFFER_LEN);
    printf( "%s>>>---pFecHugepageTbl->nPagePhyAddr=0x%lx\n",__FUNCTION__,(uint64_t)pFecHugepageTbl->nPagePhyAddr );
    mm_segment_init(&pFecSegentInfo, pFecHugepageTbl);
    printf("%s>>> pFecSegentInfo->pSegment=0x%lx, pFecSegentInfo->nPhysicalAddr=0x%lx\n", __FUNCTION__, pFecSegentInfo->pSegment, pFecSegentInfo->nPhysicalAddr);
    mm_regist_addr_to_ring(&DescRingAddr, pFecSegentInfo);

    //log init
    GLOBAL_FRLOG_INIT("/tmp/usrp/l1/", "l1", 2, 3);
    dev2_init_fec();
    dev2_register_mem_addr(&DescRingAddr);
    get_data_addr_len(&DescRingAddr, &EnDataTx, &EnDataRx, &DeDataTx, &DeDataRx);
#if 0
    // malloc memory for rx/tx data
    pEnDataSrc=(unsigned char *)malloc(DEFAULT_EN_DATA_LEN);
    pEnDataDst=(unsigned char *)malloc(DEFAULT_EN_DATA_LEN);
    // test data load
    getBinDataFromFile( pEnDataSrc, &dataLen, FecEnDataFileName );
    // int dl_encode_i;
    // for(dl_encode_i = 0; dl_encode_i<4096; dl_encode_i++){
    //   pEnDataSrc[dl_encode_i] = dl_encode_i;
    // }
    //create head;
    encode_tx_head(&EncodeHeadData);
#endif
    //saveBinDataTofile(&EncodeHeadData, sizeof(EncodeInHeaderStruct), HeadDataFileName);
    //printf("head data file save.\n");
    //sleep(100);
    // main loop !!!
    //while( loopCountEn<3 )
    #if 0
    while( 1 )
    {
        // Fec Encode
        dm_time_rec_add(&enStartTime);
        encoder_load( &EncodeHeadData, pEnDataSrc, pEnDataDst );
        dm_time_rec_add(&enEndTime);
        time_dis = dm_time_rec_diff(&enEndTime, &enStartTime);

        EnAllTime+=time_dis;
        EnAllLen+=dataLen;
        loopCountEn++;
        if(loopCountEn>=10000)
        {
            EnAllTime/=1000;
            zLog(PHY_LOG_INFO,"[FecEnRxRing] data throughput=%dMbps current time=%dus dataLen=%d ---[DRV_LOG]", 8*EnAllLen/EnAllTime/1000,time_dis, dataLen);
            loopCountEn=0;
            EnAllTime=0;
            EnAllLen=0;
        }
        printf("---end---\n");
        // for(i=0; i<10;i++){
        //     printf("%d\n",i);
        // }
        saveBinData(pEnDataDst, FEC_ENCODE_DATA);
    }
    // printf("---end---\n");
    sleep(1);
    #endif
    return 0;
}

