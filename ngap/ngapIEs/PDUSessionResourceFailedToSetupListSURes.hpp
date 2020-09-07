#ifndef _PDUSESSIONRESOURCEFAILEDTOSETUPLISTSURES_H_
#define _PDUSESSIONRESOURCEFAILEDTOSETUPLISTSURES_H_

#include "PDUSessionResourceFailedToSetupItemSURes.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceFailedToSetupListSURes.h"
}

namespace ngap{

	class PDUSessionResourceFailedToSetupListSURes{
	public:
		PDUSessionResourceFailedToSetupListSURes();
		virtual ~PDUSessionResourceFailedToSetupListSURes();
		
		void setPDUSessionResourceFailedToSetupListSURes(PDUSessionResourceFailedToSetupItemSURes *m_pduSessionResourceFailedToSetupItemSURes,int num);
		void getPDUSessionResourceFailedToSetupListSURes(PDUSessionResourceFailedToSetupItemSURes *&m_pduSessionResourceFailedToSetupItemSURes,int &num);
		
		bool encode2PDUSessionResourceFailedToSetupListSURes(Ngap_PDUSessionResourceFailedToSetupListSURes_t *pduSessionResourceFailedToSetupListSURes);
		bool decodefromPDUSessionResourceFailedToSetupListSURes(Ngap_PDUSessionResourceFailedToSetupListSURes_t *pduSessionResourceFailedToSetupListSURes);
	private:
		PDUSessionResourceFailedToSetupItemSURes *pduSessionResourceFailedToSetupItemSURes;
		int numofpduSessionResourceFailedToSetupItemSURes;
	};

}

#endif

