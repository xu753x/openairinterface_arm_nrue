#include "SecurityIndication.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	SecurityIndication::SecurityIndication()
	{
		integrityProtectionIndication = NULL;
		confidentialityProtectionIndication = NULL;
		maximumIntegrityProtectedDataRate = NULL;
	}
	SecurityIndication::~SecurityIndication(){}
	
	void SecurityIndication::setSecurityIndication(IntegrityProtectionIndication *m_integrityProtectionIndication, ConfidentialityProtectionIndication	*m_confidentialityProtectionIndication, MaximumIntegrityProtectedDataRate	*m_maximumIntegrityProtectedDataRate)
	{
		integrityProtectionIndication = m_integrityProtectionIndication;
		confidentialityProtectionIndication = m_confidentialityProtectionIndication;
		maximumIntegrityProtectedDataRate = m_maximumIntegrityProtectedDataRate;
	}
	bool SecurityIndication::encode2SecurityIndication(Ngap_SecurityIndication_t *securityIndication)
	{
		if(!integrityProtectionIndication->encode2IntegrityProtectionIndication(securityIndication->integrityProtectionIndication)) return false;
		if(!confidentialityProtectionIndication->encode2ConfidentialityProtectionIndication(securityIndication->confidentialityProtectionIndication)) return false;
		if(maximumIntegrityProtectedDataRate)
		{
			Ngap_MaximumIntegrityProtectedDataRate_t *maxIPDataRate = (Ngap_MaximumIntegrityProtectedDataRate_t *)calloc(1,sizeof(Ngap_MaximumIntegrityProtectedDataRate_t));
			if(!maxIPDataRate) return false;
			if(!maximumIntegrityProtectedDataRate->encode2MaximumIntegrityProtectedDataRate(*maxIPDataRate)) return false;
			securityIndication->maximumIntegrityProtectedDataRate = maxIPDataRate;
		}
		
		return true; 
	}
	bool SecurityIndication::decodefromSecurityIndication(Ngap_SecurityIndication_t *securityIndication)
	{
		integrityProtectionIndication = new IntegrityProtectionIndication();
		confidentialityProtectionIndication = new ConfidentialityProtectionIndication();
		
		if(!integrityProtectionIndication->decodefromIntegrityProtectionIndication(securityIndication->integrityProtectionIndication)) return false;
		if(!confidentialityProtectionIndication->decodefromConfidentialityProtectionIndication(securityIndication->confidentialityProtectionIndication)) return false;
		if(securityIndication->maximumIntegrityProtectedDataRate)
		{
			maximumIntegrityProtectedDataRate = new MaximumIntegrityProtectedDataRate();
			if(!maximumIntegrityProtectedDataRate->decodefromMaximumIntegrityProtectedDataRate(*securityIndication->maximumIntegrityProtectedDataRate)) return false;
		}
		
		return true;
	}
	void SecurityIndication::getSecurityIndication(IntegrityProtectionIndication *&m_integrityProtectionIndication, ConfidentialityProtectionIndication	*&m_confidentialityProtectionIndication, MaximumIntegrityProtectedDataRate	*&m_maximumIntegrityProtectedDataRate)
	{
		m_integrityProtectionIndication = integrityProtectionIndication;
		m_confidentialityProtectionIndication = confidentialityProtectionIndication;
		m_maximumIntegrityProtectedDataRate = maximumIntegrityProtectedDataRate;
	}

}





