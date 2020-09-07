#include "ConfidentialityProtectionResult.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	ConfidentialityProtectionResult::ConfidentialityProtectionResult()
	{
		confidentialityProtectionResult = -1;
	}  

	ConfidentialityProtectionResult::~ConfidentialityProtectionResult(){} 

	void ConfidentialityProtectionResult::setConfidentialityProtectionResult(e_Ngap_ConfidentialityProtectionResult m_confidentialityProtectionResult)
	{
		confidentialityProtectionResult = m_confidentialityProtectionResult;
	}
	bool ConfidentialityProtectionResult::getConfidentialityProtectionResult(long &m_confidentialityProtectionResult)
	{
		m_confidentialityProtectionResult = confidentialityProtectionResult;
		
		return true;
	}
	
	bool ConfidentialityProtectionResult::encode2ConfidentialityProtectionResult(Ngap_ConfidentialityProtectionResult_t &confidentialityProtectionResultValue)
	{
		confidentialityProtectionResultValue = confidentialityProtectionResult;
	
		return true;
	}
	bool ConfidentialityProtectionResult::decodefromConfidentialityProtectionResult(Ngap_ConfidentialityProtectionResult_t confidentialityProtectionResultValue)
	{
		confidentialityProtectionResult = confidentialityProtectionResultValue;
			
		return true;
	}
}


