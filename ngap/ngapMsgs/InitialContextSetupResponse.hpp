#ifndef _INITIALCONTEXTSETUPRESPONSE_H_
#define _INITIALCONTEXTSETUPRESPONSE_H_

#include "NgapIEsStruct.hpp"

#include "MessageType.hpp"
#include "AMF-UE-NGAP-ID.hpp"
#include "RAN-UE-NGAP-ID.hpp"
#include "PDUSessionResourceSetupListCxtRes.hpp"
#include "PDUSessionResourceFailedToSetupListCxtRes.hpp"

extern "C"{
	#include "Ngap_NGAP-PDU.h"
	#include "Ngap_ProtocolIE-Field.h"
	#include "Ngap_InitialContextSetupResponse.h"
}

namespace ngap{

	class InitialContextSetupResponseMsg{
	public:
		InitialContextSetupResponseMsg();
		virtual ~InitialContextSetupResponseMsg();
		
		void setMessageType();
		void setAmfUeNgapId(unsigned long id);//40 bits
		void setRanUeNgapId(uint32_t id);// 32 bits
		void setPduSessionResourceSetupResponseList(std::vector<PDUSessionResourceSetupResponseItem_t>list);
		void setPduSessionResourceFailedToSetupList(std::vector<PDUSessionResourceFailedToSetupItem_t>list);
		
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
		unsigned long getAmfUeNgapId();
		uint32_t getRanUeNgapId();
		bool getPduSessionResourceSetupResponseList(std::vector<PDUSessionResourceSetupResponseItem_t> &list);
		bool getPduSessionResourceFailedToSetupList(std::vector<PDUSessionResourceFailedToSetupItem_t> &list);

	private:
		Ngap_NGAP_PDU_t * initialContextSetupResponsePdu;
		Ngap_InitialContextSetupResponse_t *initialContextSetupResponseIEs;

		AMF_UE_NGAP_ID *amfUeNgapId;
		RAN_UE_NGAP_ID *ranUeNgapId;
		PDUSessionResourceSetupListCxtRes *pduSessionResourceSetupResponseList;
		PDUSessionResourceFailedToSetupListCxtRes *pduSessionResourceFailedToSetupResponseList;
	};

}
#endif


