#ifndef _PDUSESSIONRESOURCERELEASEITEMCMD_H_
#define _PDUSESSIONRESOURCERELEASEITEMCMD_H_

#include "PDUSessionID.hpp"
#include "NAS-PDU.hpp"
#include "S-NSSAI.hpp"
#include "PDUSessionResourceReleaseCommandTransfer.hpp"

extern "C"{
  #include "Ngap_PDUSessionResourceToReleaseItemRelCmd.h"
}

namespace ngap{

	class PDUSessionResourceReleaseItemCmd {
	public:
		PDUSessionResourceReleaseItemCmd();
		virtual ~PDUSessionResourceReleaseItemCmd();
		
		void setPDUSessionResourceReleaseItemCmd(PDUSessionID *m_pDUSessionID, OCTET_STRING_t m_pDUSessionResourceReleaseCommandTransfer);
		void getPDUSessionResourceReleaseItemCmd(PDUSessionID *&m_pDUSessionID,OCTET_STRING_t &m_pDUSessionResourceReleaseCommandTransfer);
		
		bool encode2PDUSessionResourceReleaseItemCmd(Ngap_PDUSessionResourceToReleaseItemRelCmd_t *pduSessionResourceReleaseCommandTransfer);
		bool decodefromPDUSessionResourceReleaseItemCmd(Ngap_PDUSessionResourceToReleaseItemRelCmd_t *pduSessionResourceReleaseCommandTransfer);
	private:
		PDUSessionID *pDUSessionID;
		OCTET_STRING_t pDUSessionResourceReleaseCommandTransfer;
	};
	
}

#endif





