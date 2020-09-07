#ifndef _PDUSESSIONRESOURCESETUPLISTSUREQ_H_
#define _PDUSESSIONRESOURCESETUPLISTSUREQ_H_

#include "PDUSessionResourceSetupItemSUReq.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceSetupListSUReq.h"
}

namespace ngap{

	class PDUSessionResourceSetupListSUReq{
	public:
		PDUSessionResourceSetupListSUReq();
		virtual ~PDUSessionResourceSetupListSUReq();
		
		void setPDUSessionResourceSetupListSUReq(PDUSessionResourceSetupItemSUReq *m_pduSessionResourceSetupItemSUReq,int num);
		void getPDUSessionResourceSetupListSUReq(PDUSessionResourceSetupItemSUReq *&m_pduSessionResourceSetupItemSUReq,int &num);
		
		bool encode2PDUSessionResourceSetupListSUReq(Ngap_PDUSessionResourceSetupListSUReq_t *pduSessionResourceSetupListSUReq);
		bool decodefromPDUSessionResourceSetupListSUReq(Ngap_PDUSessionResourceSetupListSUReq_t *pduSessionResourceSetupListSUReq);
	private:
		PDUSessionResourceSetupItemSUReq *pduSessionResourceSetupItemSUReq;
		int numofpduSessionResourceSetupItemSUReq;
	};

}

#endif
