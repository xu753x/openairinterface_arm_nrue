#ifndef _PDUSESSIONRESOURCESETUPLISTSURES_H_
#define _PDUSESSIONRESOURCESETUPLISTSURES_H_

#include "PDUSessionResourceSetupItemSURes.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceSetupListSURes.h"
}

namespace ngap{

	class PDUSessionResourceSetupListSURes{
	public:
		PDUSessionResourceSetupListSURes();
		virtual ~PDUSessionResourceSetupListSURes();
		
		void setPDUSessionResourceSetupListSURes(PDUSessionResourceSetupItemSURes *m_pduSessionResourceSetupItemSURes,int num);
		void getPDUSessionResourceSetupListSURes(PDUSessionResourceSetupItemSURes *&m_pduSessionResourceSetupItemSURes,int &num);
		
		bool encode2PDUSessionResourceSetupListSURes(Ngap_PDUSessionResourceSetupListSURes_t *pduSessionResourceSetupListSURes);
		bool decodefromPDUSessionResourceSetupListSURes(Ngap_PDUSessionResourceSetupListSURes_t *pduSessionResourceSetupListSURes);
	private:
		PDUSessionResourceSetupItemSURes *pduSessionResourceSetupItemSURes;
		int numofpduSessionResourceSetupItemSURes;
	};

}

#endif

