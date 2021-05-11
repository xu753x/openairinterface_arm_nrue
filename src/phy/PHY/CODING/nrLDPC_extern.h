
#include "openair1/PHY/CODING/nrLDPC_defs.h"

#ifdef LDPC_LOADER
nrLDPC_decoderfunc_t nrLDPC_decoder;
nrLDPC_encoderfunc_t nrLDPC_encoder;
#else

extern int load_nrLDPClib(void) ;
extern int load_nrLDPClib_ref(char *libversion, nrLDPC_encoderfunc_t * nrLDPC_encoder_ptr); // for ldpctest
/* ldpc coder/decoder functions, as loaded by load_nrLDPClib(). */
extern nrLDPC_decoderfunc_t nrLDPC_decoder;
extern nrLDPC_encoderfunc_t nrLDPC_encoder;
// inline functions:
#include "openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_init_mem.h"
#endif