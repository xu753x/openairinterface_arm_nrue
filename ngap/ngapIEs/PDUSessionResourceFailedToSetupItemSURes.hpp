#ifndef _PDUSESSIONRESOURCEFAILEDTOSETUPITEMSURES_H_
#define _PDUSESSIONRESOURCEFAILEDTOSETUPITEMSURES_H_

#include "PDUSessionID.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceFailedToSetupItemSURes.h"
}

namespace ngap{

	class PDUSessionResourceFailedToSetupItemSURes{
	public:
		PDUSessionResourceFailedToSetupItemSURes();
		virtual ~PDUSessionResourceFailedToSetupItemSURes();
		
		void setPDUSessionResourceFailedToSetupItemSURes(PDUSessionID *m_pDUSessionID, OCTET_STRING_t m_pDUSessionResourceSetupUnsuccessfulTransfer);
		void getPDUSessionResourceFailedToSetupItemSURes(PDUSessionID *&m_pDUSessionID, OCTET_STRING_t &m_pDUSessionResourceSetupUnsuccessfulTransfer);
		
		bool encode2PDUSessionResourceFailedToSetupItemSURes(Ngap_PDUSessionResourceFailedToSetupItemSURes_t *pduSessionResourceFailedToSetupItemSURes);
		bool decodefromPDUSessionResourceFailedToSetupItemSURes(Ngap_PDUSessionResourceFailedToSetupItemSURes_t *pduSessionResourceFailedToSetupItemSURes);
	private:
		PDUSessionID *pDUSessionID;
		OCTET_STRING_t pDUSessionResourceSetupUnsuccessfulTransfer;
	};

}

#endif
