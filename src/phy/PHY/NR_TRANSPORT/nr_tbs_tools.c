



#include "nr_transport_common_proto.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/defs_nr_common.h"

uint32_t nr_get_G(uint16_t nb_rb, uint16_t nb_symb_sch,uint8_t nb_re_dmrs,uint16_t length_dmrs, uint8_t Qm, uint8_t Nl) {
	uint32_t G;
	G = ((NR_NB_SC_PER_RB*nb_symb_sch)-(nb_re_dmrs*length_dmrs))*nb_rb*Qm*Nl;
	return(G);
}

uint32_t nr_get_E(uint32_t G, uint8_t C, uint8_t Qm, uint8_t Nl, uint8_t r) {
  uint32_t E;
  uint8_t Cprime = C; //assume CBGTI not present

  AssertFatal(Nl>0,"Nl is 0\n");
  AssertFatal(Qm>0,"Qm is 0\n");
  LOG_D(PHY,"nr_get_E : (G %d, C %d, Qm %d, Nl %d, r %d)\n",G, C, Qm, Nl, r);
  if (r <= Cprime - ((G/(Nl*Qm))%Cprime) - 1)
      E = Nl*Qm*(G/(Nl*Qm*Cprime));
  else
      E = Nl*Qm*((G/(Nl*Qm*Cprime))+1);

  return E;
}
