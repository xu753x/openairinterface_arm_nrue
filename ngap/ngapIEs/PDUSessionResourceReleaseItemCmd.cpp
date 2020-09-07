#include "PDUSessionResourceReleaseItemCmd.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceReleaseItemCmd::PDUSessionResourceReleaseItemCmd()
	{
		pDUSessionID = NULL;
	}
	PDUSessionResourceReleaseItemCmd::~PDUSessionResourceReleaseItemCmd(){}
	
	void PDUSessionResourceReleaseItemCmd::setPDUSessionResourceReleaseItemCmd(PDUSessionID *m_pDUSessionID,OCTET_STRING_t m_pDUSessionResourceReleaseCommandTransfer)
	{
		pDUSessionID = m_pDUSessionID;
		pDUSessionResourceReleaseCommandTransfer = m_pDUSessionResourceReleaseCommandTransfer;
	}
	bool PDUSessionResourceReleaseItemCmd::encode2PDUSessionResourceReleaseItemCmd(Ngap_PDUSessionResourceToReleaseItemRelCmd_t *pduSessionResourceReleaseCommandTransfer)
	{
		if(!pDUSessionID->encode2PDUSessionID(pduSessionResourceReleaseCommandTransfer->pDUSessionID)) return false;
		pduSessionResourceReleaseCommandTransfer->pDUSessionResourceReleaseCommandTransfer = pDUSessionResourceReleaseCommandTransfer;
		
		return true; 
	}
	bool PDUSessionResourceReleaseItemCmd::decodefromPDUSessionResourceReleaseItemCmd(Ngap_PDUSessionResourceToReleaseItemRelCmd_t *pduSessionResourceReleaseCommandTransfer)
	{
		pDUSessionID = new PDUSessionID();
		if(!pDUSessionID->decodefromPDUSessionID(pduSessionResourceReleaseCommandTransfer->pDUSessionID)) return false;
		pDUSessionResourceReleaseCommandTransfer = pduSessionResourceReleaseCommandTransfer->pDUSessionResourceReleaseCommandTransfer;
			
		return true;
	}
	void PDUSessionResourceReleaseItemCmd::getPDUSessionResourceReleaseItemCmd(PDUSessionID *&m_pDUSessionID, OCTET_STRING_t &m_pDUSessionResourceReleaseCommandTransfer)
	{
		m_pDUSessionID = pDUSessionID;
		m_pDUSessionResourceReleaseCommandTransfer = pDUSessionResourceReleaseCommandTransfer;
	}

}




