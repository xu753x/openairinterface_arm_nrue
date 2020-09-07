#ifndef _PDUSESSIONRESOURCEFAILEDTOSETUPLISTCXTFAIL_H_
#define _PDUSESSIONRESOURCEFAILEDTOSETUPLISTCXTFAIL_H_

#include "PDUSessionResourceFailedToSetupItemCxtFail.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceFailedToSetupListCxtFail.h"
}

namespace ngap{

	class PDUSessionResourceFailedToSetupListCxtFail{
	public:
		PDUSessionResourceFailedToSetupListCxtFail();
		virtual ~PDUSessionResourceFailedToSetupListCxtFail();
		
		void setPDUSessionResourceFailedToSetupListCxtFail(PDUSessionResourceFailedToSetupItemCxtFail *m_pduSessionResourceFailedToSetupItemCxtFail,int num);
		void getPDUSessionResourceFailedToSetupListCxtFail(PDUSessionResourceFailedToSetupItemCxtFail *&m_pduSessionResourceFailedToSetupItemCxtFail,int &num);
		
		bool encode2PDUSessionResourceFailedToSetupListCxtFail(Ngap_PDUSessionResourceFailedToSetupListCxtFail_t *pduSessionResourceFailedToSetupListCxtFail);
		bool decodefromPDUSessionResourceFailedToSetupListCxtFail(Ngap_PDUSessionResourceFailedToSetupListCxtFail_t *pduSessionResourceFailedToSetupListCxtFail);
	private:
		PDUSessionResourceFailedToSetupItemCxtFail *pduSessionResourceFailedToSetupItemCxtFail;
		int numofpduSessionResourceFailedToSetupItemCxtFail;
	};

}

#endif

