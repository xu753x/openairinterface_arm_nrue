#ifndef _PDUSESSIONRESOURCESETUPITEMSURES_H_
#define _PDUSESSIONRESOURCESETUPITEMSURES_H_

#include "PDUSessionID.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceSetupItemSURes.h"
}

namespace ngap{

	class PDUSessionResourceSetupItemSURes{
	public:
		PDUSessionResourceSetupItemSURes();
		virtual ~PDUSessionResourceSetupItemSURes();
		
		void setPDUSessionResourceSetupItemSURes(PDUSessionID *m_pDUSessionID, OCTET_STRING_t m_pDUSessionResourceSetupResponseTransfer);
		void getPDUSessionResourceSetupItemSURes(PDUSessionID *&m_pDUSessionID, OCTET_STRING_t &m_pDUSessionResourceSetupResponseTransfer);
		
		bool encode2PDUSessionResourceSetupItemSURes(Ngap_PDUSessionResourceSetupItemSURes_t *pduSessionResourceSetupItemSURes);
		bool decodefromPDUSessionResourceSetupItemSURes(Ngap_PDUSessionResourceSetupItemSURes_t *pduSessionResourceSetupItemSURes);
	private:
		PDUSessionID *pDUSessionID;
		OCTET_STRING_t pDUSessionResourceSetupResponseTransfer;
	};

}

#endif





