#include "PDUSessionResourceSetupItemCxtRes.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceSetupItemCxtRes::PDUSessionResourceSetupItemCxtRes()
	{
		pDUSessionID = NULL;
	}
	PDUSessionResourceSetupItemCxtRes::~PDUSessionResourceSetupItemCxtRes(){}
	
	void PDUSessionResourceSetupItemCxtRes::setPDUSessionResourceSetupItemCxtRes(PDUSessionID *m_pDUSessionID, OCTET_STRING_t m_pDUSessionResourceSetupResponseTransfer)
	{
		pDUSessionID = m_pDUSessionID;
		pDUSessionResourceSetupResponseTransfer = m_pDUSessionResourceSetupResponseTransfer;
	}
	bool PDUSessionResourceSetupItemCxtRes::encode2PDUSessionResourceSetupItemCxtRes(Ngap_PDUSessionResourceSetupItemCxtRes_t *pduSessionResourceSetupItemCxtRes)
	{
		if(!pDUSessionID) return false;
		if(!pDUSessionID->encode2PDUSessionID(pduSessionResourceSetupItemCxtRes->pDUSessionID)) return false;
		pduSessionResourceSetupItemCxtRes->pDUSessionResourceSetupResponseTransfer = pDUSessionResourceSetupResponseTransfer;
		
		return true; 
	}
	bool PDUSessionResourceSetupItemCxtRes::decodefromPDUSessionResourceSetupItemCxtRes(Ngap_PDUSessionResourceSetupItemCxtRes_t *pduSessionResourceSetupItemCxtRes)
	{
		pDUSessionID = new PDUSessionID();
		if(!pDUSessionID->decodefromPDUSessionID(pduSessionResourceSetupItemCxtRes->pDUSessionID)) return false;
		pDUSessionResourceSetupResponseTransfer = pduSessionResourceSetupItemCxtRes->pDUSessionResourceSetupResponseTransfer;
			
		return true;
	}
	void PDUSessionResourceSetupItemCxtRes::getPDUSessionResourceSetupItemCxtRes(PDUSessionID *&m_pDUSessionID, OCTET_STRING_t &m_pDUSessionResourceSetupResponseTransfer)
	{
		m_pDUSessionID = pDUSessionID;
		m_pDUSessionResourceSetupResponseTransfer = pDUSessionResourceSetupResponseTransfer;
	}

}




