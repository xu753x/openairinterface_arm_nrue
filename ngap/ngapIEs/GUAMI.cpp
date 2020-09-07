#include "GUAMI.hpp"

#include<iostream>
using namespace std;

namespace ngap{

	GUAMI::GUAMI()
	{
		plmnId = NULL;
		aMFRegionID = NULL;
		aMFSetID = NULL;
		aMFPointer = NULL;
	}
	GUAMI::~GUAMI(){}
	
	void GUAMI::setGUAMI(PlmnId* m_plmnId,AMFRegionID* m_aMFRegionID,AMFSetID* m_aMFSetID,AMFPointer* m_aMFPointer)
	{
		plmnId = m_plmnId;
		aMFRegionID = m_aMFRegionID;
		aMFSetID = m_aMFSetID;
		aMFPointer = m_aMFPointer;
	}
	
	bool GUAMI::encode2GUAMI(Ngap_GUAMI_t *guami)
	{
		if(!plmnId->encode2octetstring(guami->pLMNIdentity)) return false;
		if(!aMFRegionID->encode2bitstring(guami->aMFRegionID)) return false;
		if(!aMFSetID->encode2bitstring(guami->aMFSetID)) return false;
		if(!aMFPointer->encode2bitstring(guami->aMFPointer)) return false;

		return true;
	}
	bool GUAMI::decodefromGUAMI(Ngap_GUAMI_t *pdu)
	{
		plmnId = new PlmnId();
		aMFRegionID = new AMFRegionID();
		aMFSetID = new AMFSetID();
		aMFPointer = new AMFPointer();
		if(!plmnId->decodefromoctetstring(pdu->pLMNIdentity)) return false;
		if(!aMFRegionID->decodefrombitstring(pdu->aMFRegionID)) return false;
		if(!aMFSetID->decodefrombitstring(pdu->aMFSetID)) return false;
		if(!aMFPointer->decodefrombitstring(pdu->aMFPointer)) return false;

		return true;
	}
	void GUAMI::getGUAMI(PlmnId* &m_plmnId,AMFRegionID* &m_aMFRegionID,AMFSetID* &m_aMFSetID,AMFPointer* &m_aMFPointer)
	{
		m_plmnId = plmnId;
		m_aMFRegionID = aMFRegionID;
		m_aMFSetID = aMFSetID;
		m_aMFPointer = aMFPointer;
	}
}

