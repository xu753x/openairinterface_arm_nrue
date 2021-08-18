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


#include "logger_wrapper.h"
#include "ff_define.h"
#include "mem.h"


#define ADDR (void *)(0x0UL)
#define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB)
#define PROTECTION (PROT_READ | PROT_WRITE) 


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
        *segmentCount = nSegment;
        return 0;
    }

    do
    {
        if (is_contiguous (pHugepageTbl, nCount, nHugePagePerSegment, nHugePageSize))
        {

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
    }

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
    int32_t nDevFd = 0;
    void *VirtAddr = NULL;

    //segment calloc
    pSegment = (MemorySegmentStruct *)malloc (sizeof (MemorySegmentStruct) );
    if (pSegment == NULL)
    {
        printf("ERROR: No sufficient memory to store segment info\n");
        return -3;
    }
    
    *pSegInfo=pSegment;
    pSegment->nPhysicalAddr=pHugepageTbl[0].nPagePhyAddr;
    pSegment->pSegment=pHugepageTbl[0].pPageVirtAddr;
    
    //
    return 0;
}


//
//
//
//
////////////////////////////////////////////////////////
void mm_regist_addr_to_ring(DescriptorAddrRingStruct *pDescRing, MemorySegmentStruct *pFecSegInfo)
{
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
////////////////////////////////////////////////////////
void get_data_addr_len(DescriptorAddrRingStruct *pDescRingAddr, DescDataInfo *pEnDataTx, DescDataInfo *pEnDataRx, DescDataInfo *pDeDataTx, DescDataInfo *pDeDataRx)
{
    uint64_t phyAddr;

    //encode tx
    phyAddr=pDescRingAddr->TxEnCodeRingBuffer.nPagePhyAddr+RING_MEM_SIZE;
    pEnDataTx->nPhysicalAddr=phyAddr;
    pEnDataTx->nVirtAddr=pDescRingAddr->TxEnCodeRingBuffer.pPageVirtAddr+RING_MEM_SIZE;
    pEnDataTx->dataLen=0;
    //rx
    phyAddr=pDescRingAddr->RxEnCodeRingBuffer.nPagePhyAddr+RING_MEM_SIZE;
    pEnDataRx->nPhysicalAddr=phyAddr;
    pEnDataRx->nVirtAddr=pDescRingAddr->RxEnCodeRingBuffer.pPageVirtAddr+RING_MEM_SIZE;
    pEnDataRx->dataLen=0;

    //decode tx
    phyAddr=pDescRingAddr->TxDeCodeRingBuffer.nPagePhyAddr+RING_MEM_SIZE;
    pDeDataTx->nPhysicalAddr=phyAddr;
    pDeDataTx->nVirtAddr=pDescRingAddr->TxDeCodeRingBuffer.pPageVirtAddr+RING_MEM_SIZE;
    pDeDataTx->dataLen=0;
    //rx
    phyAddr=pDescRingAddr->RxDeCodeRingBuffer.nPagePhyAddr+RING_MEM_SIZE;
    pDeDataRx->nPhysicalAddr=phyAddr;
    pDeDataRx->nVirtAddr=pDescRingAddr->RxDeCodeRingBuffer.pPageVirtAddr+RING_MEM_SIZE;
    pEnDataRx->dataLen=0;

}


