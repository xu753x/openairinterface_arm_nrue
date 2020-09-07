#include "PDUSessionResourceFailedToSetupItemCxtRes.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceFailedToSetupItemCxtRes::PDUSessionResourceFailedToSetupItemCxtRes()
	{
		pDUSessionID = NULL;
	}
	PDUSessionResourceFailedToSetupItemCxtRes::~PDUSessionResourceFailedToSetupItemCxtRes(){}
	
	void PDUSessionResourceFailedToSetupItemCxtRes::setPDUSessionResourceFailedToSetupItemCxtRes(PDUSessionID *m_pDUSessionID, OCTET_STRING_t m_pDUSessionResourceSetupUnsuccessfulTransfer)
	{
		pDUSessionID = m_pDUSessionID;
		pDUSessionResourceSetupUnsuccessfulTransfer = m_pDUSessionResourceSetupUnsuccessfulTransfer;
	}
	bool PDUSessionResourceFailedToSetupItemCxtRes::encode2PDUSessionResourceFailedToSetupItemCxtRes(Ngap_PDUSessionResourceFailedToSetupItemCxtRes_t *pduSessionResourceFailedToSetupItemCxtRes)
	{
		if(!pDUSessionID) return false;
		if(!pDUSessionID->encode2PDUSessionID(pduSessionResourceFailedToSetupItemCxtRes->pDUSessionID)) return false;
		pduSessionResourceFailedToSetupItemCxtRes->pDUSessionResourceSetupUnsuccessfulTransfer = pDUSessionResourceSetupUnsuccessfulTransfer;
		
		return true; 
	}
	bool PDUSessionResourceFailedToSetupItemCxtRes::decodefromPDUSessionResourceFailedToSetupItemCxtRes(Ngap_PDUSessionResourceFailedToSetupItemCxtRes_t *pduSessionResourceFailedToSetupItemCxtRes)
	{
		pDUSessionID = new PDUSessionID();
		if(!pDUSessionID->decodefromPDUSessionID(pduSessionResourceFailedToSetupItemCxtRes->pDUSessionID)) return false;
		pDUSessionResourceSetupUnsuccessfulTransfer = pduSessionResourceFailedToSetupItemCxtRes->pDUSessionResourceSetupUnsuccessfulTransfer;
			
		return true;
	}
	void PDUSessionResourceFailedToSetupItemCxtRes::getPDUSessionResourceFailedToSetupItemCxtRes(PDUSessionID *&m_pDUSessionID, OCTET_STRING_t &m_pDUSessionResourceSetupUnsuccessfulTransfer)
	{
		m_pDUSessionID = pDUSessionID;
		pDUSessionResourceSetupUnsuccessfulTransfer = pDUSessionResourceSetupUnsuccessfulTransfer;
	}

}




