#include "SecurityResult.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	SecurityResult::SecurityResult()
	{
		integrityProtectionResult = NULL;
		confidentialityProtectionResult = NULL;
	}  

	SecurityResult::~SecurityResult(){} 

	void SecurityResult::setSecurityResult(IntegrityProtectionResult *m_integrityProtectionResult,ConfidentialityProtectionResult *m_confidentialityProtectionResult)
	{
		integrityProtectionResult = m_integrityProtectionResult;
		confidentialityProtectionResult = m_confidentialityProtectionResult;
	}
	bool SecurityResult::getSecurityResult(IntegrityProtectionResult *&m_integrityProtectionResult,ConfidentialityProtectionResult *&m_confidentialityProtectionResult)
	{
		m_integrityProtectionResult = integrityProtectionResult;
		m_confidentialityProtectionResult = confidentialityProtectionResult;
		
		return true;
	}
	
	bool SecurityResult::encode2SecurityResult(Ngap_SecurityResult_t *securityResult)
	{
		if(!integrityProtectionResult) return false;
		if(!confidentialityProtectionResult) return false;

		if(!integrityProtectionResult->encode2IntegrityProtectionResult(securityResult->integrityProtectionResult)) return false;
		if(!confidentialityProtectionResult->encode2ConfidentialityProtectionResult(securityResult->confidentialityProtectionResult)) return false;
	
		return true;
	}
	bool SecurityResult::decodefromSecurityResult(Ngap_SecurityResult_t *securityResult)
	{
		integrityProtectionResult = new IntegrityProtectionResult();
		confidentialityProtectionResult = new ConfidentialityProtectionResult();
		
		if(!integrityProtectionResult->decodefromIntegrityProtectionResult(securityResult->integrityProtectionResult)) return false;
		if(!confidentialityProtectionResult->decodefromConfidentialityProtectionResult(securityResult->confidentialityProtectionResult)) return false;
	
		return true;
	}
}
