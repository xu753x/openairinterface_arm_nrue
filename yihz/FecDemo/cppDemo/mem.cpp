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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>

extern "C"{
#include <hugetlbfs.h>
}


#include "logger_wrapper.h"
#include "ff_define.h"
#include "mem.h"
#include "nr_sys_info.h"
#include "nr_rf_chan.h"


#define ADDR (void *)(0x0UL)
#define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB)
#define PROTECTION (PROT_READ | PROT_WRITE) 

using namespace std;

//
//
//
//
////////////////////////////////////////////////////////
int get_hugepage_size(uint64_t* hugepage_size) 
{
    if (!hugepage_size) {
        return -1;
    }
    if (gethugepagesizes ((int64_t *)hugepage_size, 1) == -1) {
        printf("get huge size failed\n");
        return -1;
    }
    return 0;
}



//
//
//
//
////////////////////////////////////////////////////////
int cpa_bb_mm_virttophys (void * pVirtAddr, uint64_t * pPhysAddr)
{
    int32_t nMapFd;
    uint64_t nPage;
    uint32_t nPageSize;
    uint64_t nVirtualPageNumber;

    nMapFd = open ("/proc/self/pagemap", O_RDONLY);
    if (nMapFd < 0)
    {
        printf("ERROR: Could't open pagemap file\n");
        return 0;
    }

    /*get standard page size */
    nPageSize = (uint32_t) getpagesize ();

    nVirtualPageNumber = (uint64_t) pVirtAddr / nPageSize;

    lseek (nMapFd, nVirtualPageNumber * sizeof (uint64_t), SEEK_SET);

    if (read (nMapFd, &nPage, sizeof (uint64_t)) < 0)
    {
        close (nMapFd);
        printf("ERROR: Could't read pagemap file\n");
        return 0;
    }
    
    *pPhysAddr = ((nPage & 0x007fffffffffffffULL) * nPageSize);
    printf("nVirtualPageNumber=0x%lx, nPage=0x%lx, nPageSize=0x%x  PhysAddr=0x%lx\n", nVirtualPageNumber, nPage, nPageSize, *pPhysAddr);

    close (nMapFd);

    return 1;
}



//
//
//
//
////////////////////////////////////////////////////////
uint64_t q_partition (HugepageTblStruct * pHugepageTbl, uint64_t nLow, uint64_t nHigh)
{
    uint64_t nLowIndex = 0;
    uint64_t nHiIndex = 0;

    nLowIndex = nLow + 1;
    nHiIndex = nHigh;
    HugepageTblStruct sTemp;

    while (nHiIndex >= nLowIndex)
    {
        while ((nLowIndex <= nHiIndex) &&
         (pHugepageTbl[nLowIndex].nPagePhyAddr < pHugepageTbl[nLow].nPagePhyAddr))
        {
            nLowIndex++;
        }

        while ((nHiIndex >= nLowIndex) &&
         (pHugepageTbl[nHiIndex].nPagePhyAddr > pHugepageTbl[nLow].nPagePhyAddr))
        {
            nHiIndex--;
        }
        if (nHiIndex > nLowIndex)
        {
            memcpy (&sTemp, &pHugepageTbl[nHiIndex], sizeof (HugepageTblStruct));
            memcpy (&pHugepageTbl[nHiIndex], &pHugepageTbl[nLowIndex],
                    sizeof (HugepageTblStruct));
            memcpy (&pHugepageTbl[nLowIndex], &sTemp, sizeof (HugepageTblStruct));
        }
    }
    memcpy (&sTemp, &pHugepageTbl[nLow], sizeof (HugepageTblStruct));
    memcpy (&pHugepageTbl[nLow], &pHugepageTbl[nHiIndex], sizeof (HugepageTblStruct));
    memcpy (&pHugepageTbl[nHiIndex], &sTemp, sizeof (HugepageTblStruct));

    return nHiIndex;
}



//
//
//
//
////////////////////////////////////////////////////////
void sort_hugepage_table (HugepageTblStruct * pHugepageTbl, uint64_t nLow, uint64_t nHigh)
{
    uint64_t nKey = 0;

    if (nHigh > nLow)
    {
        nKey = q_partition (pHugepageTbl, nLow, nHigh);

        if (nKey != 0)
        {
            sort_hugepage_table (pHugepageTbl, nLow, nKey - 1);
        }

        sort_hugepage_table (pHugepageTbl, nKey + 1, nHigh);
    }

    return;
}

//
//
//
//
////////////////////////////////////////////////////////
int is_flat (uint64_t nPhysAddr1, uint64_t nPhysAddr2, uint64_t nHugePageSize)
{
    if (nPhysAddr1 > nPhysAddr2)
    {
        return 0;
    }
    else
    {
        return ((nPhysAddr1 + nHugePageSize) == nPhysAddr2) ? 1 : 0;
    }
}


//
//
//
//
////////////////////////////////////////////////////////
int is_contiguous (HugepageTblStruct * pHugepageTbl,
          uint32_t nTblOffset,
          uint32_t nHugePagePerSegment,
          uint64_t nHugePageSize)
{
    uint32_t nCount;

    for (nCount = 1; nCount < nHugePagePerSegment; nCount++)
    {
        /*Check current page and next page are contiguous */
        if (is_flat (pHugepageTbl[nTblOffset + nCount - 1].nPagePhyAddr,
          pHugepageTbl[nTblOffset + nCount].nPagePhyAddr,
          nHugePageSize) == 0)
        {
          return 0;
        }
    }
    return 1;
}


//
//
//
//
////////////////////////////////////////////////////////
int check_no_of_segments (HugepageTblStruct * pHugepageTbl,
           uint32_t nHugePage, uint32_t nSegment,
           uint32_t nHugePagePerSegment,
           uint64_t nHugePageSize, uint32_t * segmentCount)
{
    uint32_t nCount = 0;

    *segmentCount = 0;

    if (nHugePagePerSegment == 1)
    {
        for (nCount = 0; nCount < nHugePage; nCount++)
        {
          pHugepageTbl[nCount].nIsAllocated = 1;
        }
        *segmentCount = nSegment;
        return 0;
    }

    do
    {
        if (is_contiguous (pHugepageTbl, nCount, nHugePagePerSegment, nHugePageSize))
        {

            if (*segmentCount <= nSegment)
            {
              pHugepageTbl[nCount].nIsAllocated = 1;
            }

            nCount += nHugePagePerSegment;

            (*segmentCount)++;
        }
        else
        {
            nCount++;
        }
    }
    while (nCount <= (nHugePage - nHugePagePerSegment));

    printf("INFO: required segment %d segments formed %d\n",
                nSegment,
                *segmentCount);
    printf("INFO: percentage of segment got %f\n",
                (((float) *segmentCount / nSegment) * 100));

    return (*segmentCount < nSegment) ? -1 : 0;
}



//
//
//
//
////////////////////////////////////////////////////////
void mm_register_ring_addr_to_bar (DescriptorAddrRingStruct * pDring, uint64_t nFuncBaseAddr)
{
    uint32_t *pIfftReg = (uint32_t *)(nFuncBaseAddr + 0x1000);
    uint32_t *pFecDeReg =  (uint32_t *)(nFuncBaseAddr + 0x1100);
    uint32_t *pFecEnReg = (uint32_t *)(nFuncBaseAddr + 0x1200);
    uint32_t *pFftDmaReg =  (uint32_t *)(nFuncBaseAddr + 0x2000);
    uint32_t *pFecRstDeReg =  (uint32_t *)(nFuncBaseAddr + 0x2100);
    uint32_t *pFecRstEnReg =  (uint32_t *)(nFuncBaseAddr + 0x2200);
    uint32_t *pRdSyncDmaReg = (uint32_t *)(nFuncBaseAddr + 0x2300);

    uint64_t nPhysicalAddr = 0;

    pDring->BaseAddress=nFuncBaseAddr;
    //fth 
    nPhysicalAddr = pDring->TxFhRingBuffer.nPagePhyAddr;
    pIfftReg[CTRL_MODE_CFG_REG] = (BAR_MODE_READ_DESC |BAR_MODE_NOTIFY_EACH);
    pIfftReg[DESC_RING_HADDR_REG] = (uint32_t) (((uint64_t) nPhysicalAddr) >> SHIFTBITS_32);
    pIfftReg[DESC_RING_LADDR_REG] = (uint32_t) nPhysicalAddr;
    zLog(PHY_LOG_INFO, "~~ pIfftReg Reg FH Down[0x%x]:0x%lx PADDR:0x%lx ",
               0x1000, nFuncBaseAddr+0x1000, pDring->TxFhRingBuffer.nPagePhyAddr);
    nPhysicalAddr = pDring->RxFhRingBuffer.nPagePhyAddr;
    pFftDmaReg[CTRL_MODE_CFG_REG] = (BAR_MODE_READ_DESC | BAR_MODE_NOTIFY_EACH);
    pFftDmaReg[DESC_RING_HADDR_REG] = (uint32_t) (((uint64_t) nPhysicalAddr) >> SHIFTBITS_32);
    pFftDmaReg[DESC_RING_LADDR_REG] = (uint32_t) nPhysicalAddr;
    zLog(PHY_LOG_INFO, "~~ pFftDmaReg Reg FH Up  [0x%x]:0x%lx PADDR:0x%lx ",
               0x2000, nFuncBaseAddr+0x2000, pDring->RxFhRingBuffer.nPagePhyAddr);
    //fec tx 
    nPhysicalAddr = pDring->TxDeCodeRingBuffer.nPagePhyAddr;
    pFecDeReg[CTRL_MODE_CFG_REG] = (BAR_MODE_READ_DESC | BAR_MODE_NOTIFY_EACH);
    pFecDeReg[DESC_RING_HADDR_REG] = (uint32_t) (((uint64_t) nPhysicalAddr) >> SHIFTBITS_32);
    pFecDeReg[DESC_RING_LADDR_REG] = (uint32_t) nPhysicalAddr;
    zLog(PHY_LOG_INFO, "~~ pFecDeReg Reg FEC De Down[0x%x]:0x%lx PADDR:0x%lx ",
                       0x1100, nFuncBaseAddr+0x1100, pDring->TxDeCodeRingBuffer.nPagePhyAddr);
    nPhysicalAddr = pDring->TxEnCodeRingBuffer.nPagePhyAddr;
    pFecEnReg[CTRL_MODE_CFG_REG] = (BAR_MODE_READ_DESC | BAR_MODE_NOTIFY_EACH);
    pFecEnReg[DESC_RING_HADDR_REG] = (uint32_t) (((uint64_t) nPhysicalAddr) >> SHIFTBITS_32);
    pFecEnReg[DESC_RING_LADDR_REG] = (uint32_t) nPhysicalAddr;
    zLog(PHY_LOG_INFO, "~~ pFecEnReg Reg FEC En Down[0x%x]:0x%lx PADDR:0x%lx ",
                       0x1200, nFuncBaseAddr+0x1200, pDring->TxEnCodeRingBuffer.nPagePhyAddr);
    //fec rx 
    nPhysicalAddr = pDring->RxDeCodeRingBuffer.nPagePhyAddr;
    pFecRstDeReg[CTRL_MODE_CFG_REG] = (BAR_MODE_READ_DESC | BAR_MODE_NOTIFY_EACH);
    pFecRstDeReg[DESC_RING_HADDR_REG] = (uint32_t) (((uint64_t) nPhysicalAddr) >> SHIFTBITS_32);
    pFecRstDeReg[DESC_RING_LADDR_REG] = (uint32_t) nPhysicalAddr;
    zLog(PHY_LOG_INFO, "~~ pFecRstDeReg Reg FEC Rst De [0x%x]:0x%lx PADDR:0x%lx ",
                       0x2100, nFuncBaseAddr+0x2100, pDring->RxDeCodeRingBuffer.nPagePhyAddr);
    nPhysicalAddr = pDring->RxEnCodeRingBuffer.nPagePhyAddr;
    pFecRstEnReg[CTRL_MODE_CFG_REG] = (BAR_MODE_READ_DESC | BAR_MODE_NOTIFY_EACH);
    pFecRstEnReg[DESC_RING_HADDR_REG] = (uint32_t) (((uint64_t) nPhysicalAddr) >> SHIFTBITS_32);
    pFecRstEnReg[DESC_RING_LADDR_REG] = (uint32_t) nPhysicalAddr;
    zLog(PHY_LOG_INFO, "~~ pFecRstEnReg Reg FEC Rst En  [0x%x]:0x%lx PADDR:0x%lx ",
                       0x2200, nFuncBaseAddr+0x2200, pDring->RxEnCodeRingBuffer.nPagePhyAddr);
}


//
//
//
//
////////////////////////////////////////////////////////
int mm_huge_table_init(HugepageTblStruct **pHugepageTbl, uint64_t memSize)
{
    uint64_t nHugePageSize;
    uint64_t nMemorySize, nMemorySegmentSize;
    uint32_t nHugePagePerSegment = 0;
    uint32_t nSegmentCount = 0;
    uint32_t nHugePage = 0;
    uint32_t nSegment = 0;
    void *VirtAddr = NULL;
    uint64_t PhyAddr;
    uint32_t nCount = 0;
    uint32_t * pSegmentCount;
    uint32_t SegmentCount;
    HugepageTblStruct *pHT;

    if (get_hugepage_size(&nHugePageSize) != 0) {
        return -1;
    }

    nMemorySize=memSize;
    nMemorySegmentSize=SW_FPGA_SEGMENT_BUFFER_LEN;

    /*Calculate number of segment */
    nSegment = DIV_ROUND_OFF (nMemorySize, nMemorySegmentSize);

    /*Calculate number of hugepsges per segment */
    nHugePagePerSegment = DIV_ROUND_OFF (nMemorySegmentSize, nHugePageSize);

    /* calculate total number of hugepages */
    nHugePage = nSegment * nHugePagePerSegment;


    pHT = (HugepageTblStruct *) malloc (sizeof (HugepageTblStruct) * nHugePage);
    if (pHT == NULL)
    {
        printf("ERROR: HugepageTblStruct memory allocation failed cause: %s\n", strerror(errno));
        return 0;
    }

    //printf("nHugePageSize=0x%lx, nSegment=%d, nHugePagePerSegment=%d, nHugePage=%d\n", nHugePageSize, nSegment, nHugePagePerSegment, nHugePage);

    /*Allocate required number of pages to sort */
    VirtAddr = mmap (ADDR, (nHugePage * nHugePageSize), PROTECTION, FLAGS, 0, 0);
    if (VirtAddr == MAP_FAILED)
    {
        printf("ERROR: mmap was failed cause: %s\n", strerror(errno));
        free(pHT);
        return 0;
    }

    for(nCount = 0; nCount < nHugePage; nCount++)
    {
        pHT[nCount].pPageVirtAddr = (uint8_t *) VirtAddr + (nCount * nHugePageSize);
        /*Creating dummy page fault in process for each page inorder to get pagemap */
        *(uint8_t *) pHT[nCount].pPageVirtAddr = 1;
        cpa_bb_mm_virttophys(pHT[nCount].pPageVirtAddr, &pHT[nCount].nPagePhyAddr);   
        //printf("VirtAddr=0x%lx PhyAddr=0x%lx\n", (uint64_t)(uint64_t *)pHT[nCount].pPageVirtAddr, pHT[nCount].nPagePhyAddr);
        pHT[nCount].nIsAllocated = 0;
    }


    
    sort_hugepage_table(pHT, 0, nHugePage - 1);
    /*
    for(nCount = 0; nCount < nHugePage; nCount++)
    {
        printf("VirtAddr=0x%lx PhyAddr=0x%lx\n", (uint64_t)(uint64_t *)pHT[nCount].pPageVirtAddr, pHT[nCount].nPagePhyAddr);
    }
    */

    pSegmentCount=&SegmentCount;
    if (check_no_of_segments(pHT,
             nHugePage,
             nSegment,
             nHugePagePerSegment,
             nHugePageSize, pSegmentCount) == -1)
    {
        printf("ERROR: failed to get required number of pages\n");
        munmap (VirtAddr, (nHugePage * nHugePageSize));
        free (pHT);
        return -1;
    }
    pHT[0].info.nHugePage=nHugePage;
    pHT[0].info.nHugePagePerSegment=nHugePagePerSegment;
    pHT[0].info.nHugePageSize=nHugePageSize;             
    pHT[0].info.nSegment=nSegment;

    *pHugepageTbl=pHT;

    printf(">>> mm_huge_table_init is ok.\n");
    return 1;
}


//
//
//
//
////////////////////////////////////////////////////////
int mm_segment_init(MemorySegmentStruct **pSegInfo, HugepageTblStruct *pHugepageTbl)
{
    uint32_t nCount = 0;
    MemorySegmentStruct *pSegment = NULL;
    char nDevname[32];
    int32_t nDevFd = 0;
    void *VirtAddr = NULL;

    //segment calloc
    pSegment = (MemorySegmentStruct *)malloc ((sizeof (MemorySegmentStruct) * pHugepageTbl->info.nSegment ));
    if (pSegment == NULL)
    {
        printf("ERROR: No sufficient memory to store segment info\n");
        return -3;
    }
    
    sprintf(nDevname, "%s%d", DEVICE_NAME, 0);
    nDevFd = open (nDevname, O_RDWR);
    if (nDevFd < 0)
    {
        printf ("form_segments: Error opening the dev node file %s\n", strerror(errno));
        free (pSegment);
        return -3;
    }
    *pSegInfo=pSegment;
    pSegment->nPhysicalAddr=pHugepageTbl[0].nPagePhyAddr;
    pSegment->pSegment=pHugepageTbl[0].pPageVirtAddr;
/*
    //printf("*pSegInfo=%lx  pSegment=%lx\n",*pSegInfo,pSegment);
    //printf("pHugepageTbl->info.nHugePage=%d\n",pHugepageTbl->info.nHugePage);
    for (nCount = 0; nCount < pHugepageTbl->info.nHugePage; nCount++)
    {
        if (pHugepageTbl[nCount].nIsAllocated == 1)
        {
            pSegment->nSize = (pHugepageTbl->info.nHugePagePerSegment * pHugepageTbl->info.nHugePageSize);
            pSegment->pSegment = mmap (0,
                        pSegment->nSize,
                        PROT_READ | PROT_WRITE,
                        MAP_SHARED,
                        //nDevFd, pHugepageTbl[nCount].nPagePhyAddr);
                        nDevFd, 0);
            if(pSegment->pSegment == MAP_FAILED)
            {
                printf("ERROR: mmap failed in segment formation cause: %s\n", strerror(errno));
                for (; nCount > 0; nCount--)
                {
                    pSegment--;
                    munmap (pSegment->pSegment, pSegment->nSize);
                }
                free (pSegment);
                close (nDevFd);
                return -3;
            }
            *(uint8_t *)pSegment->pSegment = 1;
            usleep(100);
            printf("pSegment->nSize=0x%lx, pSegment->pSegment=0x%lx\n", pSegment->nSize, pSegment->pSegment);
            //pSegment->nPhysicalAddr=pHugepageTbl[nCount].nPagePhyAddr;
            cpa_bb_mm_virttophys(pSegment->pSegment, &pSegment->nPhysicalAddr);  
            //pSegment->nPhysicalAddr=pHugepageTbl[nCount].nPagePhyAddr+((char *)pHugepageTbl[nCount].pPageVirtAddr-(char *)pSegment->pSegment);
            //printf("pSegment->pSegment=0x%lx pSegment->nSize=%x pSegment->nPhysicalAddr=0x%lx\n",(uint64_t)(uint64_t *)pSegment->pSegment, pSegment->nSize, pSegment->nPhysicalAddr);
            pSegment++;
         }
    }
       */
    close (nDevFd);
    //
    return 0;
}


//
//
//
//
////////////////////////////////////////////////////////
void mm_regist_addr_to_ring(DescriptorAddrRingStruct *pDescRing, MemorySegmentStruct *pFthSegInfo, MemorySegmentStruct *pFecSegInfo)
{
    pDescRing->RxFhRingBuffer.pPageVirtAddr=pFthSegInfo->pSegment + FTH_MEM_SIZE*0x20;
    pDescRing->RxFhRingBuffer.nPagePhyAddr=pFthSegInfo->nPhysicalAddr + FTH_MEM_SIZE*0x20;
    pDescRing->TxFhRingBuffer.pPageVirtAddr=pFthSegInfo->pSegment+FTH_MEM_SIZE*0x40;
    pDescRing->TxFhRingBuffer.nPagePhyAddr=pFthSegInfo->nPhysicalAddr+FTH_MEM_SIZE*0x40;
    
    pDescRing->TxEnCodeRingBuffer.pPageVirtAddr=pFecSegInfo->pSegment;
    pDescRing->TxEnCodeRingBuffer.nPagePhyAddr=pFecSegInfo->nPhysicalAddr;
    pDescRing->RxEnCodeRingBuffer.pPageVirtAddr=pFecSegInfo->pSegment+FEC_MEM_SIZE*0x20;
    pDescRing->RxEnCodeRingBuffer.nPagePhyAddr=pFecSegInfo->nPhysicalAddr+FEC_MEM_SIZE*0x20;
    pDescRing->TxDeCodeRingBuffer.pPageVirtAddr=pFecSegInfo->pSegment+FEC_MEM_SIZE*0x40;
    pDescRing->TxDeCodeRingBuffer.nPagePhyAddr=pFecSegInfo->nPhysicalAddr+FEC_MEM_SIZE*0x40;
    pDescRing->RxDeCodeRingBuffer.pPageVirtAddr=pFecSegInfo->pSegment+FEC_MEM_SIZE*0x60;
    pDescRing->RxDeCodeRingBuffer.nPagePhyAddr=pFecSegInfo->nPhysicalAddr+FEC_MEM_SIZE*0x60;
}


//
//
//
//
////////////////////////////////////////////////////////
void mm_get_tti_phy_addr(MemorySegmentStruct *pFthSegInfo, uint8_t **pDmaBaseVirAddr,uint64_t *pDmaBasePhyAddr)
{
    *pDmaBaseVirAddr=(uint8_t *)pFthSegInfo->pSegment + FTH_MEM_SIZE*0x10;
    *pDmaBasePhyAddr=pFthSegInfo->nPhysicalAddr + FTH_MEM_SIZE*0x10;

    //*pDmaBaseVirAddr=(uint64_t *)pFthSegInfo->pSegment;
    //*pDmaBasePhyAddr=pFthSegInfo->nPhysicalAddr;
}


void get_data_addr_len(DescriptorAddrRingStruct *pDescRingAddr, DescDataInfo *pEnDataTx, DescDataInfo *pEnDataRx, DescDataInfo *pDeDataTx, DescDataInfo *pDeDataRx)
{
    uint64_t phyAddr;
    uint64_t nFuncBaseAddr = pDescRingAddr->BaseAddress;

    //encode tx
    phyAddr=pDescRingAddr->TxEnCodeRingBuffer.nPagePhyAddr+RING_MEM_SIZE;
    pEnDataTx->nPhysicalAddr=phyAddr;
    pEnDataTx->nVirtAddr=pDescRingAddr->TxEnCodeRingBuffer.pPageVirtAddr+RING_MEM_SIZE;
    pEnDataTx->pBarVirtAddr=(uint32_t *)(nFuncBaseAddr + 0x1200);
    pEnDataTx->dataLen=0;
    //rx
    phyAddr=pDescRingAddr->RxEnCodeRingBuffer.nPagePhyAddr+RING_MEM_SIZE;
    pEnDataRx->nPhysicalAddr=phyAddr;
    pEnDataRx->nVirtAddr=pDescRingAddr->RxEnCodeRingBuffer.pPageVirtAddr+RING_MEM_SIZE;
    pEnDataRx->pBarVirtAddr=(uint32_t *)(nFuncBaseAddr + 0x2200);
    pEnDataRx->dataLen=0;

    //decode tx
    phyAddr=pDescRingAddr->TxDeCodeRingBuffer.nPagePhyAddr+RING_MEM_SIZE;
    pDeDataTx->nPhysicalAddr=phyAddr;
    pDeDataTx->nVirtAddr=pDescRingAddr->TxDeCodeRingBuffer.pPageVirtAddr+RING_MEM_SIZE;
    pDeDataTx->pBarVirtAddr=(uint32_t *)(nFuncBaseAddr + 0x1100);
    pDeDataTx->dataLen=0;
    //rx
    phyAddr=pDescRingAddr->RxDeCodeRingBuffer.nPagePhyAddr+RING_MEM_SIZE;
    pDeDataRx->nPhysicalAddr=phyAddr;
    pDeDataRx->nVirtAddr=pDescRingAddr->RxDeCodeRingBuffer.pPageVirtAddr+RING_MEM_SIZE;
    pDeDataRx->pBarVirtAddr=(uint32_t *)(nFuncBaseAddr + 0x2100);
    pEnDataRx->dataLen=0;

}


void de_tx_update_data_addr_len(DescChainInfo *pRingInfo,DescDataInfo *pData)
{
    uint64_t phyAddr;
    uint32_t *pReg,i;

    //memset(pRingInfo, 0, sizeof(DescChainInfo));
    for(i=0;i<128;i++){
        pRingInfo->nDescriptorStatus[i] = 0;
    }

    pRingInfo->sDescriptor[0].nAddressHigh=(uint32_t) (((uint64_t) pData->nPhysicalAddr) >> SHIFTBITS_32);
    pRingInfo->sDescriptor[0].nAddressLow=(uint32_t) pData->nPhysicalAddr;
    pRingInfo->sDescriptor[0].nAddressLen=pData->dataLen;
    
    pReg=(uint32_t *)pData->pBarVirtAddr;
    pReg[VALID_DES_NUM_REG] = 1*16;
}


void de_rx_update_data_addr_len(DescChainInfo *pRingInfo,DescDataInfo *pData)
{
    uint32_t *pReg,i;

    //for(i=0;i<128;i++){
        pRingInfo->nDescriptorStatus[0] = 0;
    //}

    pRingInfo->sDescriptor[0].nAddressHigh=(uint32_t) (((uint64_t) pData->nPhysicalAddr) >> SHIFTBITS_32);
    pRingInfo->sDescriptor[0].nAddressLow=(uint32_t) pData->nPhysicalAddr;
    pReg=(uint32_t *)pData->pBarVirtAddr;
    pReg[VALID_DES_NUM_REG] = 1*16;
}



