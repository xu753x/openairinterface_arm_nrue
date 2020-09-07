#include "IntegrityProtectionIndication.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	IntegrityProtectionIndication::IntegrityProtectionIndication(){}  

	IntegrityProtectionIndication::~IntegrityProtectionIndication(){} 
	
	
	void IntegrityProtectionIndication::setIntegrityProtectionIndication(e_Ngap_IntegrityProtectionIndication m_indication)
	{
		indication = m_indication;
	}
	bool IntegrityProtectionIndication::getIntegrityProtectionIndication(long &m_indication)
	{
		m_indication = indication;
		
		return true;
	}
	
	bool IntegrityProtectionIndication::encode2IntegrityProtectionIndication(Ngap_IntegrityProtectionIndication_t &integrityProtectionIndication)
	{
		integrityProtectionIndication = indication;

		return true;
	}
	bool IntegrityProtectionIndication::decodefromIntegrityProtectionIndication(Ngap_IntegrityProtectionIndication_t integrityProtectionIndication)
	{
		indication = integrityProtectionIndication;
		
		return true;
	}

}
