



#ifndef __NR_LDPC_DEBUG__H__
#define __NR_LDPC_DEBUG__H__

#include <stdio.h>

/**
   Enum with possible LDPC data buffers
 */
typedef enum nrLDPC_buffers {
    nrLDPC_buffers_LLR_PROC, /**< LLR processing buffer */
    nrLDPC_buffers_CN_PROC, /**< CN processing buffer */
    nrLDPC_buffers_CN_PROC_RES, /**< CN processing results buffer */
    nrLDPC_buffers_BN_PROC, /**< BN processing buffer */
    nrLDPC_buffers_BN_PROC_RES, /**< BN processing results buffer */
    nrLDPC_buffers_LLR_RES /**< LLR results buffer */
} e_nrLDPC_buffers;

/**
   \brief Writes N data samples to a file
   \param fileName Name of the file
   \param p_data Pointer to the data
   \param N Number of values to write
*/
static inline void nrLDPC_writeFile(const char* fileName, int8_t* p_data, const uint32_t N)
{
    FILE *f;
    uint32_t i;

    f = fopen(fileName, "a");

    // Newline indicating new data
    fprintf(f, "\n");
    for (i=0; i < N; i++)
    {
        fprintf(f, "%d, ", p_data[i]);
    }

    fclose(f);
}

/**
   \brief Creates empty new file
   \param fileName Name of the file
*/
static inline void nrLDPC_initFile(const char* fileName)
{
    FILE *f;

    f = fopen(fileName, "w");

    fprintf(f, " ");

    fclose(f);
}

/**
   \brief Writes data of predefined buffers to file
   \param buffer Enum of buffer name to write
*/
static inline void nrLDPC_debug_writeBuffer2File(e_nrLDPC_buffers buffer, t_nrLDPC_procBuf* p_procBuf)
{
    switch (buffer)
    {
    case nrLDPC_buffers_LLR_PROC:
    {
        nrLDPC_writeFile("llrProcBuf.txt", p_procBuf->llrProcBuf, NR_LDPC_MAX_NUM_LLR);
        break;
    }
    case nrLDPC_buffers_CN_PROC:
    {
        nrLDPC_writeFile("cnProcBuf.txt", p_procBuf->cnProcBuf, NR_LDPC_SIZE_CN_PROC_BUF);
        break;
    }
    case nrLDPC_buffers_CN_PROC_RES:
    {
        nrLDPC_writeFile("cnProcBufRes.txt", p_procBuf->cnProcBufRes, NR_LDPC_SIZE_CN_PROC_BUF);
        break;
    }
    case nrLDPC_buffers_BN_PROC:
    {
        nrLDPC_writeFile("bnProcBuf.txt", p_procBuf->bnProcBuf, NR_LDPC_SIZE_BN_PROC_BUF);
        break;
    }
    case nrLDPC_buffers_BN_PROC_RES:
    {
        nrLDPC_writeFile("bnProcBufRes.txt", p_procBuf->bnProcBufRes, NR_LDPC_SIZE_BN_PROC_BUF);
        break;
    }
    case nrLDPC_buffers_LLR_RES:
    {
        nrLDPC_writeFile("llrRes.txt", p_procBuf->llrRes, NR_LDPC_MAX_NUM_LLR);
        break;
    }
    }
}

/**
   \brief Initializes file for writing a buffer
   \param buffer Enum of buffer name to write
*/
static inline void nrLDPC_debug_initBuffer2File(e_nrLDPC_buffers buffer)
{
    switch (buffer)
    {
    case nrLDPC_buffers_LLR_PROC:
    {
        nrLDPC_initFile("llrProcBuf.txt");
        break;
    }
    case nrLDPC_buffers_CN_PROC:
    {
        nrLDPC_initFile("cnProcBuf.txt");
        break;
    }
    case nrLDPC_buffers_CN_PROC_RES:
    {
        nrLDPC_initFile("cnProcBufRes.txt");
        break;
    }
    case nrLDPC_buffers_BN_PROC:
    {
        nrLDPC_initFile("bnProcBuf.txt");
        break;
    }
    case nrLDPC_buffers_BN_PROC_RES:
    {
        nrLDPC_initFile("bnProcBufRes.txt");
        break;
    }
    case nrLDPC_buffers_LLR_RES:
    {
        nrLDPC_initFile("llrRes.txt");
        break;
    }
    }
}

/**
   \brief Prints 256 data type
   \param in Input to print
*/
static inline void nrLDPC_debug_print256i_epi8(__m256i* in)
{
    uint32_t i;

    for (i=0; i<32; i++)
    {
        mexPrintf("%d ", ((int8_t*)&in)[i]);
    }
}

#endif
