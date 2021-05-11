



#ifndef __NR_LDPC_INIT_MEM__H__
#define __NR_LDPC_INIT_MEM__H__

#include <stdlib.h>
#include "nrLDPC_types.h"

#ifndef malloc32_clear
/**
   \brief Allocates 32 byte aligned memory and initializes to zero
   \param size Input size in bytes
   \return Pointer to memory
*/
static inline void* malloc32_clear(size_t size)
{
    void* ptr = (void*) memalign(32, size+32);
    memset(ptr, 0, size);
    return ptr;
}
#endif

/**
   \brief Allocates and initializes the internal decoder processing buffers
   \param p_decParams Pointer to decoder parameters
   \param p_lut Pointer to decoder LUTs
   \return Number of LLR values
*/
static inline t_nrLDPC_procBuf* nrLDPC_init_mem(void)
{
    t_nrLDPC_procBuf* p_procBuf = (t_nrLDPC_procBuf*) malloc32_clear(sizeof(t_nrLDPC_procBuf));

    if (p_procBuf)
    {
        p_procBuf->cnProcBuf    = (int8_t*) malloc32_clear(NR_LDPC_SIZE_CN_PROC_BUF*sizeof(int8_t));
        p_procBuf->cnProcBufRes = (int8_t*) malloc32_clear(NR_LDPC_SIZE_CN_PROC_BUF*sizeof(int8_t));
        p_procBuf->bnProcBuf    = (int8_t*) malloc32_clear(NR_LDPC_SIZE_BN_PROC_BUF*sizeof(int8_t));
        p_procBuf->bnProcBufRes = (int8_t*) malloc32_clear(NR_LDPC_SIZE_BN_PROC_BUF*sizeof(int8_t));
        p_procBuf->llrRes       = (int8_t*) malloc32_clear(NR_LDPC_MAX_NUM_LLR     *sizeof(int8_t));
        p_procBuf->llrProcBuf   = (int8_t*) malloc32_clear(NR_LDPC_MAX_NUM_LLR     *sizeof(int8_t));
    }

    return(p_procBuf);
}

static inline void nrLDPC_free_mem(t_nrLDPC_procBuf* p_procBuf)
{
    free(p_procBuf->cnProcBuf);
    free(p_procBuf->cnProcBufRes);
    free(p_procBuf->bnProcBuf);
    free(p_procBuf->bnProcBufRes);
    free(p_procBuf->llrRes);
    free(p_procBuf->llrProcBuf);

    free(p_procBuf);
}
#endif
