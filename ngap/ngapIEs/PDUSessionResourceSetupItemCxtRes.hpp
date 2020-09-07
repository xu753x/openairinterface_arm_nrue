#ifndef _PDUSESSIONRESOURCESETUPITEMCXTRES_H_
#define _PDUSESSIONRESOURCESETUPITEMCXTRES_H_

#include "PDUSessionID.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceSetupItemCxtRes.h"
}

namespace ngap{

	class PDUSessionResourceSetupItemCxtRes{
	public:
		PDUSessionResourceSetupItemCxtRes();
		virtual ~PDUSessionResourceSetupItemCxtRes();
		
		void setPDUSessionResourceSetupItemCxtRes(PDUSessionID *m_pDUSessionID, OCTET_STRING_t m_pDUSessionResourceSetupResponseTransfer);
		void getPDUSessionResourceSetupItemCxtRes(PDUSessionID *&m_pDUSessionID, OCTET_STRING_t &m_pDUSessionResourceSetupResponseTransfer);
		
		bool encode2PDUSessionResourceSetupItemCxtRes(Ngap_PDUSessionResourceSetupItemCxtRes_t *pduSessionResourceSetupItemCxtRes);
		bool decodefromPDUSessionResourceSetupItemCxtRes(Ngap_PDUSessionResourceSetupItemCxtRes_t *pduSessionResourceSetupItemCxtRes);
	private:
		PDUSessionID *pDUSessionID;
		OCTET_STRING_t pDUSessionResourceSetupResponseTransfer;
	};

}

#endif





