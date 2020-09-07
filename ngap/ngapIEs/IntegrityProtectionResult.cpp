#include "IntegrityProtectionResult.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	IntegrityProtectionResult::IntegrityProtectionResult()
	{
		integrityProtectionResult = -1;
	}  

	IntegrityProtectionResult::~IntegrityProtectionResult(){} 

	void IntegrityProtectionResult::setIntegrityProtectionResult(e_Ngap_IntegrityProtectionResult m_integrityProtectionResult)
	{
		integrityProtectionResult = m_integrityProtectionResult;
	}
	bool IntegrityProtectionResult::getIntegrityProtectionResult(long &m_integrityProtectionResult)
	{
		m_integrityProtectionResult = integrityProtectionResult;
		
		return true;
	}
	
	bool IntegrityProtectionResult::encode2IntegrityProtectionResult(Ngap_IntegrityProtectionResult_t &integrityProtectionResultValue)
	{
		integrityProtectionResultValue = integrityProtectionResult;
	
		return true;
	}
	bool IntegrityProtectionResult::decodefromIntegrityProtectionResult(Ngap_IntegrityProtectionResult_t integrityProtectionResultValue)
	{
		integrityProtectionResult = integrityProtectionResultValue;
			
		return true;
	}
}

