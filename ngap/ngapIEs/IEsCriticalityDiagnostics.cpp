#include "IEsCriticalityDiagnostics.hpp"

#include<iostream>
using namespace std;

namespace ngap{

	IEsCriticalityDiagnostics::IEsCriticalityDiagnostics(){}
	IEsCriticalityDiagnostics::~IEsCriticalityDiagnostics(){}
	
	void IEsCriticalityDiagnostics::setIeCriticality(Ngap_Criticality_t m_criticality)
	{
		criticality = m_criticality;
	}
	void IEsCriticalityDiagnostics::setIeId(Ngap_ProtocolIE_ID_t m_protocolIE_ID)
	{
		protocolIE_ID = m_protocolIE_ID;
	}
	void IEsCriticalityDiagnostics::setTypeOfError(Ngap_TypeOfError_t m_typeOfError)
	{
		typeOfError = m_typeOfError;
	}
	void IEsCriticalityDiagnostics::encode2pdu(Ngap_CriticalityDiagnostics_IE_Item_t * IE_Item)
	{
		IE_Item->iECriticality = criticality;
		IE_Item->iE_ID = protocolIE_ID;
		IE_Item->typeOfError = typeOfError;
	}
	
	void IEsCriticalityDiagnostics::decodefrompdu(Ngap_CriticalityDiagnostics_IE_Item_t * pdu)
	{
		criticality = pdu->iECriticality;
		protocolIE_ID = pdu->iE_ID;
		typeOfError = pdu->typeOfError;
	}
	Ngap_Criticality_t IEsCriticalityDiagnostics::getIeCriticality()
	{
		return criticality;
	}
	Ngap_ProtocolIE_ID_t IEsCriticalityDiagnostics::getIeId()
	{
		return protocolIE_ID;
	}
	Ngap_TypeOfError_t IEsCriticalityDiagnostics::getTypeOfError()
	{
		return typeOfError;
	}
}

