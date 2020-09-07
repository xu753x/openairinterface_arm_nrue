#ifndef _INITIALUEMESSAGE_H_
#define _INITIALUEMESSAGE_H_

#include "NgapIEsStruct.hpp"

#include "MessageType.hpp"
#include "RAN-UE-NGAP-ID.hpp"
#include "NAS-PDU.hpp"
#include "UserLocationInformation.hpp"
#include "RRCEstablishmentCause.hpp"
#include "UEContextRequest.hpp"
#include "FiveGSTmsi.hpp"

extern "C"{
	#include "Ngap_NGAP-PDU.h"
	#include "Ngap_ProtocolIE-Field.h"
	#include "Ngap_InitialUEMessage.h"
}

namespace ngap{

	class InitialUEMessageMsg{
	public:
		InitialUEMessageMsg();
		virtual ~InitialUEMessageMsg();

		void setMessageType();
		void setRanUENgapID(uint32_t ran_ue_ngap_id);
		void setNasPdu(uint8_t *nas,size_t sizeofnas);
		void setUserLocationInfoNR(struct NrCgi_s cig, struct Tai_s tai);
		void setRRCEstablishmentCause(e_Ngap_RRCEstablishmentCause cause_value);
		//void set5GS_TMSI(string amfSetId, string amfPointer, string _5g_tmsi);
		void setUeContextRequest(e_Ngap_UEContextRequest ueCtxReq);
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
		uint32_t getRanUENgapID();
		bool getNasPdu(uint8_t *&nas,size_t &sizeofnas);
		bool getUserLocationInfoNR(struct NrCgi_s &cig, struct Tai_s &tai);
		int  getRRCEstablishmentCause();
		int  getUeContextRequest();
                bool get5GS_TMSI(std::string &_5g_s_tmsi);

	private:
		Ngap_NGAP_PDU_t * initialUEMessagePdu;
		Ngap_InitialUEMessage_t *initialUEMessageIEs;

		RAN_UE_NGAP_ID *ranUeNgapId;
		NAS_PDU *nasPdu;
		UserLocationInformation *userLocationInformation;
		RRCEstablishmentCause *rRCEstablishmentCause;
		UEContextRequest *uEContextRequest;
                FiveGSTmsi * fivegSTmsi;
	};

}
#endif
