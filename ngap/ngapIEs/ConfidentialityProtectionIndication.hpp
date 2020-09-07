#ifndef _CONFIDENTIALITYPROTECTIONINDICATION_H_
#define _CONFIDENTIALITYPROTECTIONINDICATION_H_

extern "C"{
  #include "Ngap_ConfidentialityProtectionIndication.h"
}

namespace ngap{

	class ConfidentialityProtectionIndication{
	public:
		ConfidentialityProtectionIndication();
		virtual ~ConfidentialityProtectionIndication();
		
		void setConfidentialityProtectionIndication(e_Ngap_ConfidentialityProtectionIndication m_indication);
		bool getConfidentialityProtectionIndication(long &m_indication);
		
		bool encode2ConfidentialityProtectionIndication(Ngap_ConfidentialityProtectionIndication_t &);
		bool decodefromConfidentialityProtectionIndication(Ngap_ConfidentialityProtectionIndication_t );
	private:
		long indication;
};

}

#endif


