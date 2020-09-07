#include "ConfidentialityProtectionIndication.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	ConfidentialityProtectionIndication::ConfidentialityProtectionIndication(){}  

	ConfidentialityProtectionIndication::~ConfidentialityProtectionIndication(){} 
	
	void ConfidentialityProtectionIndication::setConfidentialityProtectionIndication(e_Ngap_ConfidentialityProtectionIndication m_indication)
	{
		indication = m_indication;
	}
	bool ConfidentialityProtectionIndication::getConfidentialityProtectionIndication(long &m_indication)
	{
		m_indication = indication;
		
		return true;
	}
	
	bool ConfidentialityProtectionIndication::encode2ConfidentialityProtectionIndication(Ngap_ConfidentialityProtectionIndication_t &confidentialityProtectionIndication)
	{
		confidentialityProtectionIndication = indication;

		return true;
	}
	bool ConfidentialityProtectionIndication::decodefromConfidentialityProtectionIndication(Ngap_ConfidentialityProtectionIndication_t confidentialityProtectionIndication)
	{
		indication = confidentialityProtectionIndication;
		
		return true;
	}

}
