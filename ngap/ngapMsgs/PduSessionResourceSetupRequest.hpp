#ifndef _PDUSESSIONRESOURCESETUPREQUEST_H_
#define _PDUSESSIONRESOURCESETUPREQUEST_H_

#include "NgapIEsStruct.hpp"

#include "MessageType.hpp"
#include "AMF-UE-NGAP-ID.hpp"
#include "RAN-UE-NGAP-ID.hpp"
#include "RANPagingPriority.hpp"
#include "NAS-PDU.hpp"
#include "PDUSessionResourceSetupListSUReq.hpp"
#include "PduSessionAggregateMaximumBitRate.hpp"

extern "C"{
	#include "Ngap_NGAP-PDU.h"
	#include "Ngap_ProtocolIE-Field.h"
	#include "Ngap_InitialContextSetupRequest.h"
}

namespace ngap{

	class PduSessionResourceSetupRequestMsg{
	public:
		PduSessionResourceSetupRequestMsg();
		virtual ~PduSessionResourceSetupRequestMsg();
		
		void setMessageType();
		void setAmfUeNgapId(unsigned long id);//40 bits
		void setRanUeNgapId(uint32_t id);// 32 bits
		void setRanPagingPriority(uint8_t priority);
		void setNasPdu(uint8_t *nas,size_t sizeofnas);
		void setPduSessionResourceSetupRequestList(std::vector<PDUSessionResourceSetupRequestItem_t>list);
		void setPduSessionAggregateMaximumBitRate(long bit_rate_downlink, long bit_rate_uplink);
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
		unsigned long getAmfUeNgapId();
		uint32_t getRanUeNgapId();
		int getRanPagingPriority();
		bool getNasPdu(uint8_t *&nas,size_t &sizeofnas);
		bool getPduSessionResourceSetupRequestList(std::vector<PDUSessionResourceSetupRequestItem_t> &list);
		//void setPduSessionAggregateMaximumBitRate(long bit_rate_downlink, long bit_rate_uplink);

	private:
		Ngap_NGAP_PDU_t * pduSessionResourceSetupRequestPdu;
		Ngap_PDUSessionResourceSetupRequest_t *pduSessionResourceSetupRequestIEs;
		
		AMF_UE_NGAP_ID *amfUeNgapId;
		RAN_UE_NGAP_ID *ranUeNgapId;
		RANPagingPriority *ranPagingPriority;
		NAS_PDU *nasPdu;
		PDUSessionResourceSetupListSUReq *pduSessionResourceSetupRequestList;
		PduSessionAggregateMaximumBitRate *pduSessionAggregateMaximumBitRate;
	};

}
#endif


