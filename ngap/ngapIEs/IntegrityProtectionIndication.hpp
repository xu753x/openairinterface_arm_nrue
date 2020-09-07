#ifndef _INTEGRITYPROTECTIONINDICATION_H_
#define _INTEGRITYPROTECTIONINDICATION_H_

extern "C"{
  #include "Ngap_IntegrityProtectionIndication.h"
}

namespace ngap{

	class IntegrityProtectionIndication{
	public:
		IntegrityProtectionIndication();
		virtual ~IntegrityProtectionIndication();
		
		void setIntegrityProtectionIndication(e_Ngap_IntegrityProtectionIndication m_indication);
		bool getIntegrityProtectionIndication(long &m_indication);
		
		bool encode2IntegrityProtectionIndication(Ngap_IntegrityProtectionIndication_t &);
		bool decodefromIntegrityProtectionIndication(Ngap_IntegrityProtectionIndication_t );
	private:
		long indication;
};

}

#endif


