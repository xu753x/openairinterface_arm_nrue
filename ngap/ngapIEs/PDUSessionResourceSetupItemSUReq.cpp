#include "PDUSessionResourceSetupItemSUReq.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceSetupItemSUReq::PDUSessionResourceSetupItemSUReq()
	{
		pDUSessionID = NULL;
		nAS_PDU = NULL;
		s_NSSAI = NULL;
	}
	PDUSessionResourceSetupItemSUReq::~PDUSessionResourceSetupItemSUReq(){}
	
	void PDUSessionResourceSetupItemSUReq::setPDUSessionResourceSetupItemSUReq(PDUSessionID *m_pDUSessionID, NAS_PDU	*m_nAS_PDU, S_NSSAI	*m_s_NSSAI,OCTET_STRING_t m_pDUSessionResourceSetupRequestTransfer)
	{
		pDUSessionID = m_pDUSessionID;
		nAS_PDU = m_nAS_PDU;
		s_NSSAI = m_s_NSSAI;
		pDUSessionResourceSetupRequestTransfer = m_pDUSessionResourceSetupRequestTransfer;
	}
	bool PDUSessionResourceSetupItemSUReq::encode2PDUSessionResourceSetupItemSUReq(Ngap_PDUSessionResourceSetupItemSUReq_t *pduSessionResourceSetupItemSUReq)
	{
		if(!pDUSessionID->encode2PDUSessionID(pduSessionResourceSetupItemSUReq->pDUSessionID)) return false;
		if(nAS_PDU)
		{
			Ngap_NAS_PDU_t	*naspdu = (Ngap_NAS_PDU_t *)calloc(1,sizeof(Ngap_NAS_PDU_t));
			if(!naspdu) return false;
			if(!nAS_PDU->encode2octetstring(*naspdu)) return false;
			pduSessionResourceSetupItemSUReq->pDUSessionNAS_PDU = naspdu;
		}
		if(!s_NSSAI->encode2S_NSSAI(&pduSessionResourceSetupItemSUReq->s_NSSAI)) return false;
		pduSessionResourceSetupItemSUReq->pDUSessionResourceSetupRequestTransfer = pDUSessionResourceSetupRequestTransfer;
		
		return true; 
	}
	bool PDUSessionResourceSetupItemSUReq::decodefromPDUSessionResourceSetupItemSUReq(Ngap_PDUSessionResourceSetupItemSUReq_t *pduSessionResourceSetupItemSUReq)
	{
		pDUSessionID = new PDUSessionID();
		s_NSSAI = new S_NSSAI();
		if(!pDUSessionID->decodefromPDUSessionID(pduSessionResourceSetupItemSUReq->pDUSessionID)) return false;
		if(!s_NSSAI->decodefromS_NSSAI(&pduSessionResourceSetupItemSUReq->s_NSSAI)) return false;
		
		if(pduSessionResourceSetupItemSUReq->pDUSessionNAS_PDU)
		{
			nAS_PDU = new NAS_PDU();
			if(!nAS_PDU->decodefromoctetstring(*pduSessionResourceSetupItemSUReq->pDUSessionNAS_PDU)) return false;
		}

		pDUSessionResourceSetupRequestTransfer = pduSessionResourceSetupItemSUReq->pDUSessionResourceSetupRequestTransfer;
			
		return true;
	}
	void PDUSessionResourceSetupItemSUReq::getPDUSessionResourceSetupItemSUReq(PDUSessionID *&m_pDUSessionID, NAS_PDU	*&m_nAS_PDU, S_NSSAI *&m_s_NSSAI,OCTET_STRING_t &m_pDUSessionResourceSetupRequestTransfer)
	{
		m_pDUSessionID = pDUSessionID;
		m_nAS_PDU = nAS_PDU;
		m_s_NSSAI = s_NSSAI;
		m_pDUSessionResourceSetupRequestTransfer = pDUSessionResourceSetupRequestTransfer;
	}

}




