#ifndef _PDUSESSIONRESOURCEFAILEDTOSETUPITEMCXTFAIL_H_
#define _PDUSESSIONRESOURCEFAILEDTOSETUPITEMCXTFAIL_H_

#include "PDUSessionID.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceFailedToSetupItemCxtFail.h"
}

namespace ngap{

	class PDUSessionResourceFailedToSetupItemCxtFail{
	public:
		PDUSessionResourceFailedToSetupItemCxtFail();
		virtual ~PDUSessionResourceFailedToSetupItemCxtFail();
		
		void setPDUSessionResourceFailedToSetupItemCxtFail(PDUSessionID *m_pDUSessionID, OCTET_STRING_t m_pDUSessionResourceSetupUnsuccessfulTransfer);
		void getPDUSessionResourceFailedToSetupItemCxtFail(PDUSessionID *&m_pDUSessionID, OCTET_STRING_t &m_pDUSessionResourceSetupUnsuccessfulTransfer);
		
		bool encode2PDUSessionResourceFailedToSetupItemCxtFail(Ngap_PDUSessionResourceFailedToSetupItemCxtFail_t *pduSessionResourceFailedToSetupItemCxtFail);
		bool decodefromPDUSessionResourceFailedToSetupItemCxtFail(Ngap_PDUSessionResourceFailedToSetupItemCxtFail_t *pduSessionResourceFailedToSetupItemCxtFail);
	private:
		PDUSessionID *pDUSessionID;
		OCTET_STRING_t pDUSessionResourceSetupUnsuccessfulTransfer;
	};

}

#endif
