#ifndef _SECURITYRESULT_H_
#define _SECURITYRESULT_H_

#include "IntegrityProtectionResult.hpp"
#include "ConfidentialityProtectionResult.hpp"

extern "C"{
  #include "Ngap_SecurityResult.h"
}

namespace ngap{

	class SecurityResult{
	public:
		SecurityResult();
		virtual ~SecurityResult();
		
		void setSecurityResult(IntegrityProtectionResult *m_integrityProtectionResult,ConfidentialityProtectionResult *m_confidentialityProtectionResult);
		bool getSecurityResult(IntegrityProtectionResult *&m_integrityProtectionResult,ConfidentialityProtectionResult *&m_confidentialityProtectionResult);
		
		bool encode2SecurityResult(Ngap_SecurityResult_t *);
		bool decodefromSecurityResult(Ngap_SecurityResult_t *);
	private:
		IntegrityProtectionResult *integrityProtectionResult;
		ConfidentialityProtectionResult *confidentialityProtectionResult;
	};

}

#endif
