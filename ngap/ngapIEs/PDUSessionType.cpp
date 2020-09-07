#include "PDUSessionType.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	PDUSessionType::PDUSessionType(){}  

	PDUSessionType::~PDUSessionType(){} 
	
	
	void PDUSessionType::setPDUSessionType(e_Ngap_PDUSessionType m_pdusessiontype)
	{
		pdusessiontype = m_pdusessiontype;
	}
	bool PDUSessionType::getPDUSessionType(long &m_pdusessiontype)
	{
		m_pdusessiontype = pdusessiontype;
		
		return true;
	}
	
	bool PDUSessionType::encode2PDUSessionType(Ngap_PDUSessionType_t &Type)
	{
		Type = pdusessiontype;

		return true;
	}
	bool PDUSessionType::decodefromPDUSessionType(Ngap_PDUSessionType_t type)
	{
		pdusessiontype = type;
		
		return true;
	}

}



