#ifndef _PDUSESSIONRESOURCESETUPLISTCXTREQ_H_
#define _PDUSESSIONRESOURCESETUPLISTCXTREQ_H_

#include "PDUSessionResourceSetupItemCxtReq.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceSetupListCxtReq.h"
}

namespace ngap{

	class PDUSessionResourceSetupListCxtReq{
	public:
		PDUSessionResourceSetupListCxtReq();
		virtual ~PDUSessionResourceSetupListCxtReq();
		
		void setPDUSessionResourceSetupListCxtReq(PDUSessionResourceSetupItemCxtReq *m_pduSessionResourceSetupItemCxtReq,int num);
		void getPDUSessionResourceSetupListCxtReq(PDUSessionResourceSetupItemCxtReq *&m_pduSessionResourceSetupItemCxtReq,int &num);
		
		bool encode2PDUSessionResourceSetupListCxtReq(Ngap_PDUSessionResourceSetupListCxtReq_t *pduSessionResourceSetupListCxtReq);
		bool decodefromPDUSessionResourceSetupListCxtReq(Ngap_PDUSessionResourceSetupListCxtReq_t *pduSessionResourceSetupListCxtReq);
	private:
		PDUSessionResourceSetupItemCxtReq *pduSessionResourceSetupItemCxtReq;
		int numofpduSessionResourceSetupItemCxtReq;
	};

}

#endif
