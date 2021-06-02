
#ifndef __FF_DEMO_DEFINE__
#define __FF_DEMO_DEFINE__

#include "fec_c_if.h"


#define FPGA_FRONTHAUL   0
#define FPGA_FEC         1
#define FPGA_NUM         2


#define IS_FREE           0
#define IS_BUSY           1
#define IS_FULL           2
#define IS_INVAL          3

#define MILLION 1000000
#define THOUSAND 1000

#define FEC_DECODE_DATA    0x01
#define FEC_ENCODE_DATA    0x02


// log mask
#define ZLOG_DRIVER_DM_TX       (1<<20)
#define ZLOG_DRIVER_DM_RX       (1<<21)
#define ZLOG_DRIVER_DM_FEC      (1<<22)
#define ZLOG_DRIVER_DM_STATUS   (1<<23)

/*
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
typedef int int32_t;
typedef long int int64_t;
typedef short int int16_t;
*/

/*
typedef struct __DescriptorAddrRingStruct
{
    uint32_t nRingBufferPoolIndex;
    HugepageTblStruct RxFhRingBuffer;
    HugepageTblStruct RxEnCodeRingBuffer;
    HugepageTblStruct RxDeCodeRingBuffer;
    HugepageTblStruct SyncFhRingBuffer;
    HugepageTblStruct TxFhRingBuffer;
    HugepageTblStruct TxEnCodeRingBuffer;
    HugepageTblStruct TxDeCodeRingBuffer;
    
    uint64_t BaseAddress;
}DescriptorAddrRingStruct;



typedef struct __DmaDescInfo
{
    uint32_t          nAddressHigh;
    uint32_t          nAddressLow;
    uint32_t          nAddressLen;
    uint32_t          rsvrd;
}DescInfo;

typedef struct __DmaDescChainInfo
{
    volatile uint32_t nDescriptorStatus[127];
    volatile uint32_t nTotalDescriptorStatus;
    DescInfo sDescriptor[127];
}DescChainInfo;


typedef struct __HugepageInfo
{
    void*  pHugepages;
    uint32_t nHugePage;
    uint32_t nSegment;
    uint32_t nHugePagePerSegment;
    uint64_t nHugePageSize;
}HugepageInfo;

typedef struct __HugepageTbl
{
    void *pPageVirtAddr;
    uint64_t nPagePhyAddr;
    uint8_t nIsAllocated:1;
    HugepageInfo info;
}HugepageTblStruct;


typedef struct __DmaDescDataInfo
{
    uint64_t nPhysicalAddr;
    void * nVirtAddr;
    void * pBarVirtAddr;
    uint32_t dataLen;
}DescDataInfo;


typedef struct __DataHeadStruct
{
    unsigned int pktType :8;
    unsigned int rsv0 :8;
    unsigned int chkCode :16;
    unsigned int pktLen :21;
    unsigned int rsv1:11;
    unsigned int rsv2;
    unsigned int rsv3;
    unsigned int rsv4;
    unsigned int rsv5;
    unsigned int rsv6;
    unsigned int rsv7;
}DataHeadInfo;
*/

typedef struct __MemorySegment
{
    void *pSegment;
    uint64_t nPhysicalAddr;
    uint32_t nSize;
    uint8_t nReferences;
}MemorySegmentStruct;




typedef union
{
    uint32_t nAll;
    struct
    {
        uint32_t nSlotIdx          : 5;
        uint32_t nSubFrameIdx      : 6;
        uint32_t nFrameIdx         : 10;
        uint32_t nSymbolIdx        : 4;
        uint32_t nReservd          : 7;
    } bit;
}TimeStempIndUnion;



#endif // __FF_DEMO_DEFINE__



