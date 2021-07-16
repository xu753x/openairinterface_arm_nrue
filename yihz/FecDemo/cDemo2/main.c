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

char FecDeDataFileName[]="rx_decode0608.bin";
// char FecDeDataFileName[]="rx_decode_matlab.bin";
// char FecDeDataFileName[]="decode_fpga.bin";
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

int decoder_load( DecodeInHeaderStruct *pHeader, unsigned char * pSrc, unsigned char * pDst, unsigned char * pCRC )
{
    DecodeInHeaderStruct *pDataHeadInfo;
    int InHeaderLength = sizeof(DecodeInHeaderStruct);
    int DDRHeaderLength = ((pHeader->cbNum+7)/8)*32;
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
            memcpy( (unsigned char *)DeDataTx.nVirtAddr, pHeader, InHeaderLength);
            memcpy( (unsigned char *)DeDataTx.nVirtAddr+InHeaderLength+DDRHeaderLength, pSrc, pHeader->pktLen-InHeaderLength-DDRHeaderLength);
            DeDataTx.dataLen=pHeader->pktLen;
            dev2_send_de_data(&DeDataTx);
            zLog(PHY_LOG_INFO," decoder_load:<1> ---[DRV_LOG]");
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
            zLog(PHY_LOG_INFO," decoder_load:<3.5> ---[DRV_LOG]");
            // printf("fec_de_rx_release_start\n");
            pDataHeadInfo=(DecodeOutHeaderStruct *)DeDataRx.nVirtAddr;
            // memcpy(pDst, (unsigned char *)(DeDataRx.nVirtAddr), (pDataHeadInfo->tbSizeB));
            memcpy(pDst, (unsigned char *)(DeDataRx.nVirtAddr+32), (pDataHeadInfo->tbSizeB));
            memcpy(pCRC, (unsigned char *)( DeDataRx.nVirtAddr+32 + (pDataHeadInfo->tbSizeB+(4-pDataHeadInfo->tbSizeB%4)%4) ), (4));
            if(*pCRC == 1)
            {
                zLog(PHY_LOG_INFO," decoder_load: crc correct!");
                // memcpy(pDst, (unsigned char *)(DeDataRx.nVirtAddr+32), (pDataHeadInfo->tbSizeB-32));
            }
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
#if 0 //最先测试的头部参数，对应rx_decode_matlab
    //word 0
    pHD->pktType=0x10;
    pHD->rsv0=0x00;
    pHD->chkCode=0xFAFA;
    //word 1
    pHD->pktLen=0x68A0;
    pHD->rsv1=0x0000;
    //word 2
    pHD->pktTpTmp=0x0;
    pHD->pduSize=0x1A28;
    pHD->sectorId=0x0;
    pHD->rsv2=0x0;
    //word 3
    pHD->sfn=0x121;
    pHD->rsv3=0x0;
    pHD->subfn=0x04;
    pHD->slotNum=0x08;
    pHD->pduIdx=0x0;
    pHD->rev4=0x0;
    //word 4
    pHD->tbSizeB=0x8C1;
    pHD->rev5=0x0;
    pHD->lastTb=0x1;
    pHD->firstTb=0x1;
    pHD->rev6=0x0;
    pHD->cbNum=0x03;
    //word 5
    pHD->qm=0x01;
    pHD->rev7=0x0;
    pHD->fillbit=0x148;
    pHD->kpInByte=0x1778;
    //word 6
    pHD->gamma=0x03;
    pHD->maxRowNm=0x2E;
    pHD->maxRvIdx=0x0;
    pHD->rvIdx=0x0;
    pHD->ndi=0x01;
    pHD->flush=0x0;
    pHD->maxIter=0x04;
    pHD->lfSizeIx=0x05;
    pHD->rev10=0x0;
    pHD->iLs=0x04;
    pHD->bg=0x0;
    //word 7
    pHD->e1=0x22C8;
    pHD->e0=0x22C8;
#endif

#if 1   //最新测试的头部参数，对应rx_decode0608
    //word 0
    pHD->pktType=16;
    pHD->rsv0=0x00;
    pHD->chkCode=64250;
    //word 1
    pHD->pktLen=38464;
    pHD->rsv1=0x0000;
    //word 2
    pHD->pktTpTmp=0;
    pHD->pduSize=9616;
    pHD->sectorId=0;
    pHD->rsv2=0x0;
    //word 3
    pHD->sfn=199;
    pHD->rsv3=0x0;
    pHD->subfn=4;
    pHD->slotNum=8;
    pHD->pduIdx=0;
    pHD->rev4=0x0;
    //word 4
    pHD->tbSizeB=1761;
    pHD->rev5=0x0;
    pHD->lastTb=1;
    pHD->firstTb=1;
    pHD->rev6=0x0;
    pHD->cbNum=2;
    //word 5
    pHD->qm=2;
    pHD->rev7=0x0;
    pHD->fillbit=664;
    pHD->kpInByte=7080;
    //word 6
    pHD->gamma=2;
    pHD->maxRowNm=46;
    pHD->maxRvIdx=0;
    pHD->rvIdx=0;
    pHD->ndi=1;
    pHD->flush=0;
    pHD->maxIter=4;
    pHD->lfSizeIx=5;
    pHD->rev10=0x0;
    pHD->iLs=5;
    pHD->bg=0;
    //word 7
    pHD->e1=19200;
    pHD->e0=19200;
#endif
}

//
//
//
//
////////////////////////////////////////////////////////
#if 0   //encode的demo的main
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
    // DecodeInHeaderStruct DecodeHeadData;
    unsigned char *pEnDataSrc,*pEnDataDst;
    // unsigned char *pDeDataSrc,*pDeDataDst;
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
    // pDeDataSrc=(unsigned char *)malloc(DEFAULT_DE_DATA_LEN);
    // pDeDataDst=(unsigned char *)malloc(DEFAULT_DE_DATA_LEN);
    // test data load
    getBinDataFromFile( pEnDataSrc, &dataLen, FecEnDataFileName );
    // getBinDataFromFile( pDeDataSrc, &dataLen, FecDeDataFileName );
    //create head;
    encode_tx_head(&EncodeHeadData);
    // decode_tx_head(&DecodeHeadData);

    //saveBinDataTofile(&EncodeHeadData, sizeof(EncodeInHeaderStruct), HeadDataFileName);
    //printf("head data file save.\n");
    //sleep(100);
    // main loop !!!
    //while( loopCountEn<3 )
    // while( 1 )
    // {
        // Fec Encode
        dm_time_rec_add(&enStartTime);
        // dm_time_rec_add(&deStartTime);
        encoder_load( &EncodeHeadData, pEnDataSrc, pEnDataDst );
        // decoder_load( &DecodeHeadData, pDeDataSrc, pDeDataDst );
        dm_time_rec_add(&enEndTime);
        // dm_time_rec_add(&deEndTime);
        time_dis = dm_time_rec_diff(&enEndTime, &enStartTime);
        // time_dis = dm_time_rec_diff(&deEndTime, &deStartTime);

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
        // DeAllTime+=time_dis;
        // DeAllLen+=dataLen;
        // loopCountDe++;
        // if(loopCountDe>=10000)
        // {
        //     DeAllTime/=1000;
        //     //zLog(PHY_LOG_INFO,"[FecEnRxRing] data throughput=%dMbps current time=%dus dataLen=%d ---[DRV_LOG]", 8*EnAllLen/EnAllTime/1000,time_dis, dataLen);
        //     loopCountDe=0;
        //     DeAllTime=0;
        //     DeAllLen=0;
        // }
        printf("---end---\n");
        // for(i=0; i<10;i++){
        //     printf("%d\n",i);
        // }
        for(i=0; i<30; i++)
        {
            printf("pEnDataDst+%d = %d\n", i, *(pEnDataDst+i));
        }
        saveBinData(pEnDataDst, FEC_ENCODE_DATA);
        // saveBinData(pDeDataDst, FEC_DECODE_DATA);
    // }
    // printf("---end---\n");
    // sleep(1);
    return 0;
}
#endif

#if 1   //decode的demo的main
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
    DecodeInHeaderStruct DecodeHeadData;
    unsigned char *pEnDataSrc,*pEnDataDst;
    char *pDeDataSrc;
    unsigned char *pDeDataDst;
    int dataLen=0;
    unsigned char *pcrc;

    
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
    // pEnDataSrc=(unsigned char *)malloc(DEFAULT_EN_DATA_LEN);
    // pEnDataDst=(unsigned char *)malloc(DEFAULT_EN_DATA_LEN);
    pDeDataSrc=(char *)malloc(DEFAULT_DE_DATA_LEN);
    pDeDataDst=(unsigned char *)malloc(DEFAULT_DE_DATA_LEN);
    pcrc = (unsigned char *)malloc(4);
    // test data load
    getBinDataFromFile( pDeDataSrc, &dataLen, FecDeDataFileName );
    
    //create head;
    // encode_tx_head(&EncodeHeadData);
    decode_tx_head(&DecodeHeadData);

    //saveBinDataTofile(&EncodeHeadData, sizeof(EncodeInHeaderStruct), HeadDataFileName);
    //printf("head data file save.\n");
    //sleep(100);
    // main loop !!!
    //while( loopCountEn<3 )
    // while( 1 )
    // {
        // Fec Encode
        dm_time_rec_add(&deStartTime);
        // encoder_load( &EncodeHeadData, pEnDataSrc, pEnDataDst );
        //测试的demo中pDeDataSrc包含了32Byte的头，应该去掉。
        // decoder_load( &DecodeHeadData, pDeDataSrc, pDeDataDst );
        decoder_load( &DecodeHeadData, pDeDataSrc+64, pDeDataDst, pcrc );
        dm_time_rec_add(&deEndTime);
        time_dis = dm_time_rec_diff(&deEndTime, &deStartTime);

        DeAllTime+=time_dis;
        DeAllLen+=dataLen;
        loopCountDe++;
        if(loopCountDe>=10000)
        {
            DeAllTime/=1000;
            zLog(PHY_LOG_INFO,"[FecDeRxRing] data throughput=%dMbps current time=%dus dataLen=%d ---[DRV_LOG]", 8*DeAllLen/DeAllTime/1000,time_dis, dataLen);
            loopCountDe=0;
            DeAllTime=0;
            DeAllLen=0;
        }
        printf("---end---\n");
        // for(i=0; i<10;i++){
        //     printf("%d\n",i);
        // }
        for(i=0; i<100; i++)
        {
            printf("pDeDataSrc+%d = %d\n", i, *(pDeDataSrc+i));
        }
        printf("--------------------------------------\n");
        // for(i=0; i<DecodeHeadData.tbSizeB+8; i++)
        // {
        //     printf("pDeDataDst+%d = %u\n", i, *(pDeDataDst+i));
        // }
        for(i=0; i<200; i++)
        {
            printf("pDeDataDst+%d = %u\n", i, *(pDeDataDst+i));
        }
        for(i=0; i<4; i++)
        {
            printf("pcrc+%d = %u\n", i, *(pcrc+i));
        }
        if(*pcrc == 1)
        {
            printf("decode correct!\n");
        }
        saveBinData(pDeDataDst, FEC_DECODE_DATA);
    // }
    // printf("---end---\n");
    // sleep(1);
    return 0;
}
#endif

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

int main234(int aaa)
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
    DecodeInHeaderStruct DecodeHeadData;
    unsigned char *pEnDataSrc,*pEnDataDst;
    char *pDeDataSrc;
    unsigned char *pDeDataDst;
    int dataLen=0;
    unsigned char *pcrc;

    
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
    // pEnDataSrc=(unsigned char *)malloc(DEFAULT_EN_DATA_LEN);
    // pEnDataDst=(unsigned char *)malloc(DEFAULT_EN_DATA_LEN);
    pDeDataSrc=(char *)malloc(DEFAULT_DE_DATA_LEN);
    pDeDataDst=(unsigned char *)malloc(DEFAULT_DE_DATA_LEN);
    pcrc = (unsigned char *)malloc(4);
    // test data load
    getBinDataFromFile( pDeDataSrc, &dataLen, FecDeDataFileName );
    
    //create head;
    // encode_tx_head(&EncodeHeadData);
    decode_tx_head(&DecodeHeadData);

    //saveBinDataTofile(&EncodeHeadData, sizeof(EncodeInHeaderStruct), HeadDataFileName);
    //printf("head data file save.\n");
    //sleep(100);
    // main loop !!!
    //while( loopCountEn<3 )
    // while( 1 )
    // {
        // Fec Encode
        dm_time_rec_add(&deStartTime);
        // encoder_load( &EncodeHeadData, pEnDataSrc, pEnDataDst );
        //测试的demo中pDeDataSrc包含了32Byte的头，应该去掉。
        // decoder_load( &DecodeHeadData, pDeDataSrc, pDeDataDst );
        decoder_load( &DecodeHeadData, pDeDataSrc+32, pDeDataDst, pcrc );
        dm_time_rec_add(&deEndTime);
        time_dis = dm_time_rec_diff(&deEndTime, &deStartTime);

        DeAllTime+=time_dis;
        DeAllLen+=dataLen;
        loopCountDe++;
        if(loopCountDe>=10000)
        {
            DeAllTime/=1000;
            zLog(PHY_LOG_INFO,"[FecDeRxRing] data throughput=%dMbps current time=%dus dataLen=%d ---[DRV_LOG]", 8*DeAllLen/DeAllTime/1000,time_dis, dataLen);
            loopCountDe=0;
            DeAllTime=0;
            DeAllLen=0;
        }
        printf("---end---\n");
        // for(i=0; i<10;i++){
        //     printf("%d\n",i);
        // }
        for(i=0; i<100; i++)
        {
            printf("pDeDataSrc+%d = %d\n", i, *(pDeDataSrc+i));
        }
        printf("--------------------------------------\n");
        // for(i=0; i<DecodeHeadData.tbSizeB+8; i++)
        // {
        //     printf("pDeDataDst+%d = %u\n", i, *(pDeDataDst+i));
        // }
        for(i=0; i<200; i++)
        {
            printf("pDeDataDst+%d = %u\n", i, *(pDeDataDst+i));
        }
        for(i=0; i<4; i++)
        {
            printf("pcrc+%d = %u\n", i, *(pcrc+i));
        }
        if(*pcrc == 1)
        {
            printf("decode correct!\n");
        }
        saveBinData(pDeDataDst, FEC_DECODE_DATA);
    // }
    // printf("---end---\n");
    // sleep(1);
    return 0;
}