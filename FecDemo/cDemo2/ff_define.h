
#ifndef __FF_DEMO_DEFINE__
#define __FF_DEMO_DEFINE__



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

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
typedef int int32_t;
typedef long int int64_t;
typedef short int int16_t;

#include <dev2.0/fec_c_if.h>


typedef struct
{
    /*Word 0*/
    uint32_t pktType  :8;
    uint32_t rsv0     :8;
    uint32_t chkCode  :16;

    /*Word 1*/
    uint32_t pktLen   :21;    /*按照byte计算，header+tbsize     按照byte对齐后的长度*/
    uint32_t rsv1     :11;

    /*Word 2*/
    uint32_t rsv2     :28;
    uint32_t sectorId :2;
    uint32_t rsv3     :2;

    /*Word 3*/
    uint32_t sfn      :10;
    uint32_t rsv4     :2;
    uint32_t subfn    :4;
    uint32_t slotNum  :5;
    uint32_t pduIdx   :9;
    uint32_t rev5     :2;
    
    /*Word 4*/
	uint32_t tbSizeB  :18;
    uint32_t rev6     :2;
    uint32_t lastTb   :1;
    uint32_t firstTb  :1;
    uint32_t rev7     :2;
    uint32_t cbNum    :8;
    
    /*Word 5*/
    uint32_t qm       :3;
    uint32_t rev8     :1;
    uint32_t fillbit  :14;
    uint32_t rev9     :2;
    uint32_t kpInByte :11;
    uint32_t rev10    :1;
    
    /*Word 6*/
    uint32_t gamma    :8;
    uint32_t codeRate :6;
    uint32_t rev11    :2;
    uint32_t rvIdx    :2;
    uint32_t rev12    :6;
    uint32_t lfSizeIx :3;
    uint32_t rev13    :1;
    uint32_t iLs      :3;
    uint32_t bg       :1;
    
    /*Word 7*/
    uint32_t e1       :16;
    uint32_t e0       :16;
}EncodeInHeaderStruct;

typedef struct
{
    uint32_t pktType  :8;
    uint32_t rsv0     :8;
    uint32_t chkCode  :16;

    uint32_t pduSize  :21;
    uint32_t rsv1     :11;

    uint32_t rsv2     :28;
    uint32_t sectorId :2;
    uint32_t rsv3     :2;

    uint32_t sfn      :10;
    uint32_t rsv4     :2;
    uint32_t subfn    :4;
    uint32_t slotNum  :5;
    uint32_t pduIdx   :9;
    uint32_t rev5     :2;
	
    uint32_t tbSizeB  :18;
    uint32_t rev6     :2;
    uint32_t lastTb   :1;
    uint32_t firstTb  :1;
    uint32_t rev7     :2;
    uint32_t cbNum    :8;

    uint32_t qm       :3;
    uint32_t rev8     :1;
    uint32_t fillbit  :14;
    uint32_t rev9     :2;
    uint32_t kpInByte :11;
    uint32_t rev10    :1;

    uint32_t gamma    :8;
    uint32_t codeRate :6;
    uint32_t rev11    :2;
    uint32_t rvIdx    :2;
    uint32_t rev12    :6;
    uint32_t lfSizeIx :3;
    uint32_t rev13    :1;
    uint32_t iLs      :3;
    uint32_t bg       :1;

    uint32_t e1       :16;
    uint32_t e0       :16;
}EncodeOutHeaderStruct;

typedef struct
{
    /*Word 0*/
    uint32_t pktType  :8;
    uint32_t rsv0     :8;
    uint32_t chkCode  :16;

    /*Word 1*/
    uint32_t pktLen   :21;    /*包含header + DDRheader + CB DATA的长度,单位是byte*/
    uint32_t rsv1     :11;

    /*Word 2*/
    uint32_t pktTpTmp :4;
    uint32_t pduSize  :24;    /*包含header + DDRheader + CB DATA的长度,单位是Word*/
    uint32_t sectorId :2;
    uint32_t rsv2     :2;

    /*Word 3*/
    uint32_t sfn      :10;
    uint32_t rsv3     :2;
    uint32_t subfn    :4;
    uint32_t slotNum  :5;
    uint32_t pduIdx   :9;
    uint32_t rev4     :2;

    /*Word 4*/
    uint32_t tbSizeB  :18;   /*tbsize的大小，单位byte*/
    uint32_t rev5     :2;
    uint32_t lastTb   :1;
    uint32_t firstTb  :1;
    uint32_t rev6     :2;
    uint32_t cbNum    :8;    /*码块数*/

    /*Word 5*/
    uint32_t qm       :3;
    uint32_t rev7     :1;
    uint32_t fillbit  :14;
    uint32_t kpInByte :14;   /* 被均分后每个cb块的长度，单位bit */

    /*Word 6*/
    uint32_t gamma    :8;
    uint32_t maxRowNm :6;    //填成46
    uint32_t maxRvIdx :2;
    uint32_t rvIdx    :2;
    uint32_t ndi      :1;
    uint32_t flush    :1;
    uint32_t maxIter  :4;
    uint32_t lfSizeIx :3;
    uint32_t rev10    :1;
    uint32_t iLs      :3;
    uint32_t bg       :1;

    /*Word 7*/
    uint32_t e1       :16;
    uint32_t e0       :16;
}DecodeInHeaderStruct;

typedef struct
{
    /*word 0*/
    uint32_t pktType  :8;   /* FEC RX 类型： 0x11*/
    uint32_t rsv0     :8;
    uint32_t chkCode  :16;  /* 数据校验 ：0xFAFA*/

    /*word 1*/
    uint32_t pktLen   :21;  /* 包括 FPGA_ALIGN(header+ FPGA_ALIGN4B(tbsizeB)+4byte) */
    uint32_t rsv1     :11;

    /*word 2*/
    uint32_t pktTpTmp :4;
    uint32_t pduSize  :20;  /* pktLen长度除以4 单位 word*/
    uint32_t rsv2     :4;
    uint32_t sectorId :2;
    uint32_t rsv3     :2;

    /*word 3*/
    uint32_t sfn      :10;  /* 帧号 */
    uint32_t rsv4     :2;
    uint32_t subfn    :4;   /* 子帧号 */
    uint32_t slotNum  :5;   /* 时隙号 */
    uint32_t pduIdx   :9;
    uint32_t rsv5     :2;

    /*Word 4*/
    uint32_t tbSizeB  :18;  /* tbsize 的大小，单位byte*/
    uint32_t rsv6     :2;
    uint32_t lastTb   :1;
    uint32_t firstTb  :1;
    uint32_t rsv7     :2;
    uint32_t cbNum    :8;

    /*Word 5*/
    uint32_t rsv8     :32;

    /*Word 6*/
    uint32_t rsv9     :32;

    /*Word 7*/
    uint32_t rsv10    :32;
}DecodeOutHeaderStruct;


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



