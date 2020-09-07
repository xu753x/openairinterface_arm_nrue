#include "PDUSessionResourceFailedToSetupItemSURes.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceFailedToSetupItemSURes::PDUSessionResourceFailedToSetupItemSURes()
	{
		pDUSessionID = NULL;
	}
	PDUSessionResourceFailedToSetupItemSURes::~PDUSessionResourceFailedToSetupItemSURes(){}
	
	void PDUSessionResourceFailedToSetupItemSURes::setPDUSessionResourceFailedToSetupItemSURes(PDUSessionID *m_pDUSessionID, OCTET_STRING_t m_pDUSessionResourceSetupUnsuccessfulTransfer)
	{
		pDUSessionID = m_pDUSessionID;
		pDUSessionResourceSetupUnsuccessfulTransfer = m_pDUSessionResourceSetupUnsuccessfulTransfer;
	}
	bool PDUSessionResourceFailedToSetupItemSURes::encode2PDUSessionResourceFailedToSetupItemSURes(Ngap_PDUSessionResourceFailedToSetupItemSURes_t *pduSessionResourceFailedToSetupItemSURes)
	{
		if(!pDUSessionID) return false;
		if(!pDUSessionID->encode2PDUSessionID(pduSessionResourceFailedToSetupItemSURes->pDUSessionID)) return false;
		pduSessionResourceFailedToSetupItemSURes->pDUSessionResourceSetupUnsuccessfulTransfer = pDUSessionResourceSetupUnsuccessfulTransfer;
		
		return true; 
	}
	bool PDUSessionResourceFailedToSetupItemSURes::decodefromPDUSessionResourceFailedToSetupItemSURes(Ngap_PDUSessionResourceFailedToSetupItemSURes_t *pduSessionResourceFailedToSetupItemSURes)
	{
		pDUSessionID = new PDUSessionID();
		if(!pDUSessionID->decodefromPDUSessionID(pduSessionResourceFailedToSetupItemSURes->pDUSessionID)) return false;
		pDUSessionResourceSetupUnsuccessfulTransfer = pduSessionResourceFailedToSetupItemSURes->pDUSessionResourceSetupUnsuccessfulTransfer;
			
		return true;
	}
	void PDUSessionResourceFailedToSetupItemSURes::getPDUSessionResourceFailedToSetupItemSURes(PDUSessionID *&m_pDUSessionID, OCTET_STRING_t &m_pDUSessionResourceSetupUnsuccessfulTransfer)
	{
		m_pDUSessionID = pDUSessionID;
		m_pDUSessionResourceSetupUnsuccessfulTransfer = pDUSessionResourceSetupUnsuccessfulTransfer;
	}

}




