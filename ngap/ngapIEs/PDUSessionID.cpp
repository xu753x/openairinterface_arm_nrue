#include "PDUSessionID.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	PDUSessionID::PDUSessionID(){}  

	PDUSessionID::~PDUSessionID(){} 
	
	
	void PDUSessionID::setPDUSessionID(uint8_t m_pdusessionid)
	{
		pdusessionid = m_pdusessionid;
	}
	bool PDUSessionID::getPDUSessionID(uint8_t &m_pdusessionid)
	{
		m_pdusessionid = pdusessionid;
		
		return true;
	}
	
	bool PDUSessionID::encode2PDUSessionID(Ngap_PDUSessionID_t &pduSessionId)
	{
		pduSessionId = pdusessionid;

		return true;
	}
	bool PDUSessionID::decodefromPDUSessionID(Ngap_PDUSessionID_t pduSessionId)
	{
		pdusessionid = pduSessionId;
		
		return true;
	}

}


