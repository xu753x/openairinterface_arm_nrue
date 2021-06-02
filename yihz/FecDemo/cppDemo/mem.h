#ifndef __DEV2_FF_DEMO_MEM__
#define __DEV2_FF_DEMO_MEM__


/*DMA CONTROL REG*/
#define CTRL_MODE_CFG_REG      0
#define STATUS_HADDR_REG       1
#define STATUS_LADDR_REG       2
#define VALID_DES_NUM_REG      3
#define DESC_RING_HADDR_REG    4
#define DESC_RING_LADDR_REG    5

#define BAR_MODE_READ_DESC      0x0  /*2'b00'*/
#define BAR_MODE_READ_MEM       0x2  /*2'b10'*/
#define BAR_MODE_NOTIFY_EACH    0x0  /*2'b00'*/
#define BAR_MODE_NOTIFY_ALL     0x1  /*2'b01'*/


/* hugepage setting */
#define SW_FPGA_TOTAL_BUFFER_LEN 4LL*1024*1024*1024
#define SW_FPGA_SEGMENT_BUFFER_LEN 1LL*1024*1024*1024
#define SW_FPGA_FH_TOTAL_BUFFER_LEN 1LL*1024*1024*1024

#define DIV_ROUND_OFF(X,Y) ( X/Y + ((X%Y)?1:0) )
#define DEVICE_NAME "/dev/nr_cdev"

#define FTH_MEM_SIZE    (0x100000)
#define FEC_MEM_SIZE    (0x100000)
#define RING_MEM_SIZE    (0x10000)
#define SHIFTBITS_32                (32)
#define IS_DESC_DONE(desc) (0xFF&desc)

//
int mm_huge_table_init(HugepageTblStruct **pHugepageTbl, uint64_t memSize);
//
int mm_segment_init(MemorySegmentStruct **pFecSegInfo, HugepageTblStruct *pHugepageTbl);
//
void mm_regist_addr_to_ring(DescriptorAddrRingStruct *pDescRing, MemorySegmentStruct *pFthSegInfo, MemorySegmentStruct *pFecSegInfo);
//
void mm_get_tti_phy_addr(MemorySegmentStruct *pFthSegInfo, uint8_t **pDmaBaseVirAddr,uint64_t *pDmaBasePhyAddr);
//
void mm_register_ring_addr_to_bar ( DescriptorAddrRingStruct * pDring, uint64_t nFuncBaseAddr);
//
void de_tx_update_data_addr_len(DescChainInfo *pRingInfo,DescDataInfo *pData);
//
void de_rx_update_data_addr_len(DescChainInfo *pRingInfo,DescDataInfo *pData);
//
void get_data_addr_len(DescriptorAddrRingStruct *pDescRingAddr, DescDataInfo *pEnDataTx, DescDataInfo *pEnDataRx, DescDataInfo *pDeDataTx, DescDataInfo *pDeDataRx);


#endif  //__DEV2_FF_DEMO_MEM__
