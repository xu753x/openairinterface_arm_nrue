#include "PDUSessionResourceSetupItemSURes.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceSetupItemSURes::PDUSessionResourceSetupItemSURes()
	{
		pDUSessionID = NULL;
	}
	PDUSessionResourceSetupItemSURes::~PDUSessionResourceSetupItemSURes(){}
	
	void PDUSessionResourceSetupItemSURes::setPDUSessionResourceSetupItemSURes(PDUSessionID *m_pDUSessionID, OCTET_STRING_t m_pDUSessionResourceSetupResponseTransfer)
	{
		pDUSessionID = m_pDUSessionID;
		pDUSessionResourceSetupResponseTransfer = m_pDUSessionResourceSetupResponseTransfer;
	}
	bool PDUSessionResourceSetupItemSURes::encode2PDUSessionResourceSetupItemSURes(Ngap_PDUSessionResourceSetupItemSURes_t *pduSessionResourceSetupItemSURes)
	{
		if(!pDUSessionID) return false;
		if(!pDUSessionID->encode2PDUSessionID(pduSessionResourceSetupItemSURes->pDUSessionID)) return false;
		pduSessionResourceSetupItemSURes->pDUSessionResourceSetupResponseTransfer = pDUSessionResourceSetupResponseTransfer;
		
		return true; 
	}
	bool PDUSessionResourceSetupItemSURes::decodefromPDUSessionResourceSetupItemSURes(Ngap_PDUSessionResourceSetupItemSURes_t *pduSessionResourceSetupItemSURes)
	{
		pDUSessionID = new PDUSessionID();
		if(!pDUSessionID->decodefromPDUSessionID(pduSessionResourceSetupItemSURes->pDUSessionID)) return false;
		pDUSessionResourceSetupResponseTransfer = pduSessionResourceSetupItemSURes->pDUSessionResourceSetupResponseTransfer;
			
		return true;
	}
	void PDUSessionResourceSetupItemSURes::getPDUSessionResourceSetupItemSURes(PDUSessionID *&m_pDUSessionID, OCTET_STRING_t &m_pDUSessionResourceSetupResponseTransfer)
	{
		m_pDUSessionID = pDUSessionID;
		m_pDUSessionResourceSetupResponseTransfer = pDUSessionResourceSetupResponseTransfer;
	}

}




