#ifndef _PDUSESSIONRESOURCESETUPITEMCXTREQ_H_
#define _PDUSESSIONRESOURCESETUPITEMCXTREQ_H_

#include "PDUSessionID.hpp"
#include "NAS-PDU.hpp"
#include "S-NSSAI.hpp"
//#include "PDUSessionResourceSetupRequestTransfer.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceSetupItemCxtReq.h"
}

namespace ngap{

	class PDUSessionResourceSetupItemCxtReq{
	public:
		PDUSessionResourceSetupItemCxtReq();
		virtual ~PDUSessionResourceSetupItemCxtReq();
		
		void setPDUSessionResourceSetupItemCxtReq(PDUSessionID *m_pDUSessionID, NAS_PDU	*m_nAS_PDU, S_NSSAI	*m_s_NSSAI,OCTET_STRING_t m_pDUSessionResourceSetupRequestTransfer);
		void getPDUSessionResourceSetupItemCxtReq(PDUSessionID *&m_pDUSessionID, NAS_PDU	*&m_nAS_PDU, S_NSSAI *&m_s_NSSAI,OCTET_STRING_t &m_pDUSessionResourceSetupRequestTransfer);
		
		bool encode2PDUSessionResourceSetupItemCxtReq(Ngap_PDUSessionResourceSetupItemCxtReq_t *pduSessionResourceSetupItemCxtReq);
		bool decodefromPDUSessionResourceSetupItemCxtReq(Ngap_PDUSessionResourceSetupItemCxtReq_t *pduSessionResourceSetupItemCxtReq);
	private:
		PDUSessionID *pDUSessionID;
		NAS_PDU	*nAS_PDU;	/* OPTIONAL */
		S_NSSAI	 *s_NSSAI;
		//PduSessionResourceSetupRequestTransferIE *pduSessionResourceSetupRequestTransfer;
		OCTET_STRING_t pDUSessionResourceSetupRequestTransfer;
	};

}

#endif





