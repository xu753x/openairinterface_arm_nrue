#ifndef _UPLINKNASTRANSPORT_H_
#define _UPLINKNASTRANSPORT_H_

#include "NgapIEsStruct.hpp"

#include "MessageType.hpp"
#include "AMF-UE-NGAP-ID.hpp"
#include "RAN-UE-NGAP-ID.hpp"
#include "NAS-PDU.hpp"
#include "UserLocationInformation.hpp"

extern "C"{
	#include "Ngap_NGAP-PDU.h"
	#include "Ngap_ProtocolIE-Field.h"
	#include "Ngap_InitialUEMessage.h"
}

namespace ngap{

	class UplinkNASTransportMsg{
	public:
		UplinkNASTransportMsg();
		virtual ~UplinkNASTransportMsg();

		void setMessageType();
		
		void setAmfUeNgapId(unsigned long id);//40 bits
		void setRanUeNgapId(uint32_t id);// 32 bits
		void setNasPdu(uint8_t *nas,size_t sizeofnas);
		void setUserLocationInfoNR(struct NrCgi_s cig, struct Tai_s tai);
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
		unsigned long getAmfUeNgapId();
		uint32_t getRanUeNgapId();
		bool getNasPdu(uint8_t *&nas,size_t &sizeofnas);
		bool getUserLocationInfoNR(struct NrCgi_s &cig, struct Tai_s &tai);

	private:
		Ngap_NGAP_PDU_t * uplinkNASTransportPdu;
		Ngap_UplinkNASTransport_t *uplinkNASTransportIEs;

		AMF_UE_NGAP_ID *amfUeNgapId;
		RAN_UE_NGAP_ID *ranUeNgapId;
		NAS_PDU *nasPdu;
		UserLocationInformation *userLocationInformation;
	};

}
#endif
