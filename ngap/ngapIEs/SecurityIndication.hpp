#ifndef _SECURITYINDICATION_H_
#define _SECURITYINDICATION_H_

#include "IntegrityProtectionIndication.hpp"
#include "ConfidentialityProtectionIndication.hpp"
#include "MaximumIntegrityProtectedDataRate.hpp"

extern "C"{
  #include "Ngap_SecurityIndication.h"
}

namespace ngap{

	class SecurityIndication{
	public:
		SecurityIndication();
		virtual ~SecurityIndication();
		
		void setSecurityIndication(IntegrityProtectionIndication *m_integrityProtectionIndication, ConfidentialityProtectionIndication	*m_confidentialityProtectionIndication, MaximumIntegrityProtectedDataRate	*m_maximumIntegrityProtectedDataRate);
		void getSecurityIndication(IntegrityProtectionIndication *&m_integrityProtectionIndication, ConfidentialityProtectionIndication	*&m_confidentialityProtectionIndication, MaximumIntegrityProtectedDataRate	*&m_maximumIntegrityProtectedDataRate);
		
		bool encode2SecurityIndication(Ngap_SecurityIndication_t *securityIndication);
		bool decodefromSecurityIndication(Ngap_SecurityIndication_t *securityIndication);
	private:
		IntegrityProtectionIndication *integrityProtectionIndication;
		ConfidentialityProtectionIndication	*confidentialityProtectionIndication;
		MaximumIntegrityProtectedDataRate *maximumIntegrityProtectedDataRate;
};

}

#endif
