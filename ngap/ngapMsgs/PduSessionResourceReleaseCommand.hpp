#ifndef _PDUSESSIONRESOURCERELEASECOMMAND_H_
#define _PDUSESSIONRESOURCERELEASECOMMAND_H_

#include "NgapIEsStruct.hpp"

#include "MessageType.hpp"
#include "AMF-UE-NGAP-ID.hpp"
#include "RAN-UE-NGAP-ID.hpp"
#include "PDUSessionResourceReleaseListCmd.hpp"

extern "C"{
	#include "Ngap_NGAP-PDU.h"
	#include "Ngap_ProtocolIE-Field.h"
	#include "Ngap_PDUSessionResourceReleaseCommand.h"
}

namespace ngap{

	class PduSessionResourceReleaseCommand {
	public:
		PduSessionResourceReleaseCommand();
		virtual ~PduSessionResourceReleaseCommand();
		
		void setMessageType();
		void setAmfUeNgapId(unsigned long id);//40 bits
		void setRanUeNgapId(uint32_t id);// 32 bits
		void setPduSessionResourceToReleaseList(std::vector<PDUSessionResourceReleaseCommandItem_t>list);
	
		
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
		unsigned long getAmfUeNgapId();
		uint32_t getRanUeNgapId();
		bool getPduSessionResourceToReleaseList(std::vector<PDUSessionResourceReleaseCommandItem_t> &list);
		

	private:
		Ngap_NGAP_PDU_t * pduSessionResourceReleaseCommandPdu;
		Ngap_PDUSessionResourceReleaseCommand_t *pduSessionResourceReleaseCommandIEs;

		AMF_UE_NGAP_ID *amfUeNgapId;
		RAN_UE_NGAP_ID *ranUeNgapId;
		PDUSessionResourceReleaseListCmd *pduSessionResourceReleaseCommandList;

	};

}
#endif


