#ifndef _PDUSESSIONRESOURCERELEASELISTCMD_H_
#define _PDUSESSIONRESOURCERELEASELISTCMD_H_

#include "PDUSessionResourceReleaseItemCmd.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceToReleaseListRelCmd.h"
}

namespace ngap{

	class PDUSessionResourceReleaseListCmd {
	public:
		PDUSessionResourceReleaseListCmd();
		virtual ~PDUSessionResourceReleaseListCmd();
		
		void setPDUSessionResourceReleaseListCmd(PDUSessionResourceReleaseItemCmd *m_pduSessionResourceReleaseItemCmd,int num);
		void getPDUSessionResourceReleaseListCmd(PDUSessionResourceReleaseItemCmd *&m_pduSessionResourceReleaseItemCmd,int &num);
		
		bool encode2PDUSessionResourceReleaseListCmd(Ngap_PDUSessionResourceToReleaseListRelCmd_t *pduSessionResourceReleaseListCmd);
		bool decodefromPDUSessionResourceReleaseListCmd(Ngap_PDUSessionResourceToReleaseListRelCmd_t *pduSessionResourceReleaseListCmd);
	private:
		PDUSessionResourceReleaseItemCmd *pduSessionResourceReleaseItemCmd;
		int numofpduSessionResourceReleaseItemCmd;
	};

}

#endif

