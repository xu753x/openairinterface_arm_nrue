#ifndef _PDUSESSIONRESOURCESETUPITEMSUREQ_H_
#define _PDUSESSIONRESOURCESETUPITEMSUREQ_H_

#include "PDUSessionID.hpp"
#include "NAS-PDU.hpp"
#include "S-NSSAI.hpp"
//#include "PDUSessionResourceSetupRequestTransfer.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceSetupItemSUReq.h"
}

namespace ngap{

	class PDUSessionResourceSetupItemSUReq{
	public:
		PDUSessionResourceSetupItemSUReq();
		virtual ~PDUSessionResourceSetupItemSUReq();
		
		void setPDUSessionResourceSetupItemSUReq(PDUSessionID *m_pDUSessionID, NAS_PDU	*m_nAS_PDU, S_NSSAI	*m_s_NSSAI,OCTET_STRING_t m_pDUSessionResourceSetupRequestTransfer);
		void getPDUSessionResourceSetupItemSUReq(PDUSessionID *&m_pDUSessionID, NAS_PDU	*&m_nAS_PDU, S_NSSAI *&m_s_NSSAI,OCTET_STRING_t &m_pDUSessionResourceSetupRequestTransfer);
		
		bool encode2PDUSessionResourceSetupItemSUReq(Ngap_PDUSessionResourceSetupItemSUReq_t *pduSessionResourceSetupItemSUReq);
		bool decodefromPDUSessionResourceSetupItemSUReq(Ngap_PDUSessionResourceSetupItemSUReq_t *pduSessionResourceSetupItemSUReq);
	private:
		PDUSessionID *pDUSessionID;
		NAS_PDU	*nAS_PDU;	/* OPTIONAL */
		S_NSSAI	 *s_NSSAI;
		//PduSessionResourceSetupRequestTransferIE *pduSessionResourceSetupRequestTransfer;
		OCTET_STRING_t pDUSessionResourceSetupRequestTransfer;
	};

}

#endif





