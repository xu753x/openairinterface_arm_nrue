#ifndef _ALLOWEDNSSAI_H_
#define _ALLOWEDNSSAI_H_

#include "S-NSSAI.hpp"

extern "C"{
  #include "Ngap_AllowedNSSAI.h"
}

namespace ngap{

	class AllowedNSSAI{
	public:
		AllowedNSSAI();
		virtual ~AllowedNSSAI();
		
		void setAllowedNSSAI(S_NSSAI * m_snssai, int m_numofsnssai);
		bool getAllowedNSSAI(S_NSSAI * & m_snssai, int & m_numofsnssai);
		bool encode2AllowedNSSAI(Ngap_AllowedNSSAI_t *allowedNssaiList);
		bool decodefromAllowedNSSAI(Ngap_AllowedNSSAI_t *allowedNssaiList);
	private:
		S_NSSAI * snssai;
		int numofSnssai;
	};
}
#endif

