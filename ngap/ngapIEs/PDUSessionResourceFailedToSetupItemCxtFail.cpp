#include "PDUSessionResourceFailedToSetupItemCxtFail.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceFailedToSetupItemCxtFail::PDUSessionResourceFailedToSetupItemCxtFail()
	{
		pDUSessionID = NULL;
	}
	PDUSessionResourceFailedToSetupItemCxtFail::~PDUSessionResourceFailedToSetupItemCxtFail(){}
	
	void PDUSessionResourceFailedToSetupItemCxtFail::setPDUSessionResourceFailedToSetupItemCxtFail(PDUSessionID *m_pDUSessionID, OCTET_STRING_t m_pDUSessionResourceSetupUnsuccessfulTransfer)
	{
		pDUSessionID = m_pDUSessionID;
		pDUSessionResourceSetupUnsuccessfulTransfer = m_pDUSessionResourceSetupUnsuccessfulTransfer;
	}
	bool PDUSessionResourceFailedToSetupItemCxtFail::encode2PDUSessionResourceFailedToSetupItemCxtFail(Ngap_PDUSessionResourceFailedToSetupItemCxtFail_t *pduSessionResourceFailedToSetupItemCxtFail)
	{
		if(!pDUSessionID) return false;
		if(!pDUSessionID->encode2PDUSessionID(pduSessionResourceFailedToSetupItemCxtFail->pDUSessionID)) return false;
		pduSessionResourceFailedToSetupItemCxtFail->pDUSessionResourceSetupUnsuccessfulTransfer = pDUSessionResourceSetupUnsuccessfulTransfer;
		
		return true; 
	}
	bool PDUSessionResourceFailedToSetupItemCxtFail::decodefromPDUSessionResourceFailedToSetupItemCxtFail(Ngap_PDUSessionResourceFailedToSetupItemCxtFail_t *pduSessionResourceFailedToSetupItemCxtFail)
	{
		pDUSessionID = new PDUSessionID();
		if(!pDUSessionID->decodefromPDUSessionID(pduSessionResourceFailedToSetupItemCxtFail->pDUSessionID)) return false;
		pDUSessionResourceSetupUnsuccessfulTransfer = pduSessionResourceFailedToSetupItemCxtFail->pDUSessionResourceSetupUnsuccessfulTransfer;
			
		return true;
	}
	void PDUSessionResourceFailedToSetupItemCxtFail::getPDUSessionResourceFailedToSetupItemCxtFail(PDUSessionID *&m_pDUSessionID, OCTET_STRING_t &m_pDUSessionResourceSetupUnsuccessfulTransfer)
	{
		m_pDUSessionID = pDUSessionID;
		pDUSessionResourceSetupUnsuccessfulTransfer = pDUSessionResourceSetupUnsuccessfulTransfer;
	}

}




