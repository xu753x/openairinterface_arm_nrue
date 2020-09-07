#ifndef _INITIALCONTEXTSETUPFAILURE_H_
#define _INITIALCONTEXTSETUPFAILURE_H_

#include "NgapIEsStruct.hpp"

#include "MessageType.hpp"
#include "AMF-UE-NGAP-ID.hpp"
#include "RAN-UE-NGAP-ID.hpp"
#include "PDUSessionResourceFailedToSetupListCxtFail.hpp"

extern "C"{
	#include "Ngap_NGAP-PDU.h"
	#include "Ngap_ProtocolIE-Field.h"
	#include "Ngap_InitialContextSetupFailure.h"
}

namespace ngap{

	class InitialContextSetupFailureMsg{
	public:
		InitialContextSetupFailureMsg();
		virtual ~InitialContextSetupFailureMsg();
		
		void setMessageType();
		void setAmfUeNgapId(unsigned long id);//40 bits
		void setRanUeNgapId(uint32_t id);// 32 bits
		void setPduSessionResourceFailedToSetupList(std::vector<PDUSessionResourceFailedToSetupItem_t>list);
		
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
		unsigned long getAmfUeNgapId();
		uint32_t getRanUeNgapId();
		bool getPduSessionResourceFailedToSetupList(std::vector<PDUSessionResourceFailedToSetupItem_t> &list);

	private:
		Ngap_NGAP_PDU_t * initialContextSetupFailurePdu;
		Ngap_InitialContextSetupFailure_t *initialContextSetupFailureIEs;

		AMF_UE_NGAP_ID *amfUeNgapId;
		RAN_UE_NGAP_ID *ranUeNgapId;
		PDUSessionResourceFailedToSetupListCxtFail *pduSessionResourceFailedToSetupFailureList;
	};


}
#endif

