#ifndef _PDUSESSIONRESOURCEFAILEDTOSETUPITEMCXTRES_H_
#define _PDUSESSIONRESOURCEFAILEDTOSETUPITEMCXTRES_H_

#include "PDUSessionID.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceFailedToSetupItemCxtRes.h"
}

namespace ngap{

	class PDUSessionResourceFailedToSetupItemCxtRes{
	public:
		PDUSessionResourceFailedToSetupItemCxtRes();
		virtual ~PDUSessionResourceFailedToSetupItemCxtRes();
		
		void setPDUSessionResourceFailedToSetupItemCxtRes(PDUSessionID *m_pDUSessionID, OCTET_STRING_t m_pDUSessionResourceSetupUnsuccessfulTransfer);
		void getPDUSessionResourceFailedToSetupItemCxtRes(PDUSessionID *&m_pDUSessionID, OCTET_STRING_t &m_pDUSessionResourceSetupUnsuccessfulTransfer);
		
		bool encode2PDUSessionResourceFailedToSetupItemCxtRes(Ngap_PDUSessionResourceFailedToSetupItemCxtRes_t *pduSessionResourceFailedToSetupItemCxtRes);
		bool decodefromPDUSessionResourceFailedToSetupItemCxtRes(Ngap_PDUSessionResourceFailedToSetupItemCxtRes_t *pduSessionResourceFailedToSetupItemCxtRes);
	private:
		PDUSessionID *pDUSessionID;
		OCTET_STRING_t pDUSessionResourceSetupUnsuccessfulTransfer;
	};

}

#endif
