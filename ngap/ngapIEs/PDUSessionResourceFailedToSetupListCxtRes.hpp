#ifndef _PDUSESSIONRESOURCEFAILEDTOSETUPLISTCXTRES_H_
#define _PDUSESSIONRESOURCEFAILEDTOSETUPLISTCXTRES_H_

#include "PDUSessionResourceFailedToSetupItemCxtRes.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceFailedToSetupListCxtRes.h"
}

namespace ngap{

	class PDUSessionResourceFailedToSetupListCxtRes{
	public:
		PDUSessionResourceFailedToSetupListCxtRes();
		virtual ~PDUSessionResourceFailedToSetupListCxtRes();
		
		void setPDUSessionResourceFailedToSetupListCxtRes(PDUSessionResourceFailedToSetupItemCxtRes *m_pduSessionResourceFailedToSetupItemCxtRes,int num);
		void getPDUSessionResourceFailedToSetupListCxtRes(PDUSessionResourceFailedToSetupItemCxtRes *&m_pduSessionResourceFailedToSetupItemCxtRes,int &num);
		
		bool encode2PDUSessionResourceFailedToSetupListCxtRes(Ngap_PDUSessionResourceFailedToSetupListCxtRes_t *pduSessionResourceFailedToSetupListCxtRes);
		bool decodefromPDUSessionResourceFailedToSetupListCxtRes(Ngap_PDUSessionResourceFailedToSetupListCxtRes_t *pduSessionResourceFailedToSetupListCxtRes);
	private:
		PDUSessionResourceFailedToSetupItemCxtRes *pduSessionResourceFailedToSetupItemCxtRes;
		int numofpduSessionResourceFailedToSetupItemCxtRes;
	};

}

#endif

