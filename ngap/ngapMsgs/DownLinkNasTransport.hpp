#ifndef _DOWNLINKNASTRANSPORT_H_
#define _DOWNLINKNASTRANSPORT_H_

#include "MessageType.hpp"
#include "AMF-UE-NGAP-ID.hpp"
#include "RAN-UE-NGAP-ID.hpp"
#include "AMFName.hpp"
#include "RANPagingPriority.hpp"
#include "NAS-PDU.hpp"
#include "IndexToRFSP.hpp"

extern "C"{
	#include "Ngap_NGAP-PDU.h"
	#include "Ngap_ProtocolIE-Field.h"
	#include "Ngap_DownlinkNASTransport.h"
}

namespace ngap{

	class DownLinkNasTransportMsg{
	public:
		DownLinkNasTransportMsg();
		virtual ~DownLinkNasTransportMsg();

		void setMessageType();
		void setAmfUeNgapId(unsigned long id);//40 bits
		void setRanUeNgapId(uint32_t id);// 32 bits
		void setOldAmfName(const std::string name);
		void setRanPagingPriority(uint8_t pagingPriority);//1~256
		void setNasPdu(uint8_t *nas,size_t sizeofnas);
		void setIndex2Rat_Frequency_SelectionPriority(uint8_t value);//1~256
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
		unsigned long getAmfUeNgapId();
		uint32_t getRanUeNgapId();
		bool getOldAmfName(std::string &name);
		uint8_t getRanPagingPriority();
		bool getNasPdu(uint8_t *&nas,size_t &sizeofnas);
		uint8_t getIndex2Rat_Frequency_SelectionPriority();

	private:
		Ngap_NGAP_PDU_t * downLinkNasTransportPdu;
		Ngap_DownlinkNASTransport_t *downLinkNasTransportIEs;

		AMF_UE_NGAP_ID *amfUeNgapId;
		RAN_UE_NGAP_ID *ranUeNgapId;
		AmfName *oldAmfName;
		RANPagingPriority *ranPagingPriority;
		NAS_PDU *nasPdu;
		IndexToRFSP *indexToRFSP;
	};

}
#endif

