#ifndef _PDUSESSIONRESOURCESETUPLISTCXTRES_H_
#define _PDUSESSIONRESOURCESETUPLISTCXTRES_H_

#include "PDUSessionResourceSetupItemCxtRes.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceSetupListCxtRes.h"
}

namespace ngap{

	class PDUSessionResourceSetupListCxtRes{
	public:
		PDUSessionResourceSetupListCxtRes();
		virtual ~PDUSessionResourceSetupListCxtRes();
		
		void setPDUSessionResourceSetupListCxtRes(PDUSessionResourceSetupItemCxtRes *m_pduSessionResourceSetupItemCxtRes,int num);
		void getPDUSessionResourceSetupListCxtRes(PDUSessionResourceSetupItemCxtRes *&m_pduSessionResourceSetupItemCxtRes,int &num);
		
		bool encode2PDUSessionResourceSetupListCxtRes(Ngap_PDUSessionResourceSetupListCxtRes_t *pduSessionResourceSetupListCxtRes);
		bool decodefromPDUSessionResourceSetupListCxtRes(Ngap_PDUSessionResourceSetupListCxtRes_t *pduSessionResourceSetupListCxtRes);
	private:
		PDUSessionResourceSetupItemCxtRes *pduSessionResourceSetupItemCxtRes;
		int numofpduSessionResourceSetupItemCxtRes;
	};

}

#endif

