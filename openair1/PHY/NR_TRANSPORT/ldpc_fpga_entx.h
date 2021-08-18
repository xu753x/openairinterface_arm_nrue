//FPGA LDPC加速接口头部定义
typedef struct{
    /*Word 0*/
    uint32_t pktType  :8; //0x12
    uint32_t rsv0     :8; //空
    uint32_t chkCode  :16; //0xFAFA
    /*Word 1*/
    uint32_t pktLen   :21;    //Byte，pktLen=encoder header(32byte)+ tbszie (byte)，并且32Byte对齐，是32的整数倍
    uint32_t rsv1     :11;
    /*Word 2*/
    uint32_t rsv2     :28;
    uint32_t sectorId :2; //=0表示单小区
    uint32_t rsv3     :2;
    /*Word 3*/
    uint32_t sfn      :10; //每次都不一样
    uint32_t rsv4     :2;
    uint32_t subfn    :4; //=slotNum/2
    uint32_t slotNum  :5;
    uint32_t pduIdx   :9; //=0表示第一个码字，总共一个码字
    uint32_t rev5     :2;
    /*Word 4*/
    uint32_t tbSizeB  :18;
    uint32_t rev6     :2;
    uint32_t lastTb   :1; //=1表示本slot只有一个TB
    uint32_t firstTb  :1; //=1表示本slot只有一个TB
    uint32_t rev7     :2;
    uint32_t cbNum    :8;
    /*Word 5*/
    uint32_t qm       :3; //规定是BPSK qm=0,QPSK qm=1,其他floor(调制阶数/2)
    uint32_t rev8     :1;
    uint32_t fillbit  :14;
    uint32_t rev9     :2;
    uint32_t kpInByte :11;  //encode的单位是Byte
    uint32_t rev10    :1;
    /*Word 6*/
    uint32_t gamma    :8;
    uint32_t codeRate :6; //For LDPC base graph 1, a matrix of   has 46 rows ，For LDPC base graph 2, a matrix of   has 42 rows(看bg)
    uint32_t rev11    :2;
    uint32_t rvIdx    :2;
    uint32_t rev12    :6;
    uint32_t lfSizeIx :3; //由lfSizeIx（列）和iLs（行）找到Zc
    uint32_t rev13    :1;
    uint32_t iLs      :3;
    uint32_t bg       :1; //规定选择协议base grape1 bg=0; base grape2 bg=1
    /*Word 7*/
    uint32_t e1       :16;
    uint32_t e0       :16;
}EncodeInHeadStruct;

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
}DecodeInHeadStruct;

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
}DecodeOutHeadStruct;
