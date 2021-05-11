#ifndef __NR_LDPC_DECODER_LYC__H__
#define __NR_LDPC_DECODER_LYC__H__



 



int32_t nrLDPC_decoder_LYC(t_nrLDPC_dec_params* p_decParams, int8_t* p_llr, int8_t* p_out, int block_length, time_stats_t *time_decoder);


void init_LLR_DMA_for_CUDA(t_nrLDPC_dec_params* p_decParams, int8_t* p_llr, int8_t* p_out, int block_length);

void warmup_for_GPU(void);

void set_compact_BG(int Zc, short BG);


#endif
