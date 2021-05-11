
//============================================================================================================================
// encoder interface
#ifndef __NRLDPC_DEFS__H__
#define __NRLDPC_DEFS__H__
#include "openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_types.h"

typedef struct {
	int n_segments;          // optim8seg
	unsigned int macro_num; // optim8segmulti
	unsigned char gen_code; //orig
	time_stats_t *tinput;
	time_stats_t *tprep;
	time_stats_t *tparity;
	time_stats_t *toutput;
}encoder_implemparams_t;
#define INIT0_LDPCIMPLEMPARAMS {0,0,0,NULL,NULL,NULL,NULL}
typedef int(*nrLDPC_encoderfunc_t)(unsigned char **,unsigned char **,int,int,short, short, encoder_implemparams_t*);
//============================================================================================================================
// decoder interface
/**
   \brief LDPC decoder API type definition
   \param p_decParams LDPC decoder parameters
   \param p_llr Input LLRs
   \param p_llrOut Output vector
   \param p_profiler LDPC profiler statistics
*/
typedef int32_t(*nrLDPC_decoderfunc_t)(t_nrLDPC_dec_params* , int8_t*, int8_t* , t_nrLDPC_procBuf* , t_nrLDPC_time_stats* );
#endif