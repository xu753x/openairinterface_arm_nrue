#ifndef _UERADIOCAPABILITYINFOINDICATION_H_
#define _UERADIOCAPABILITYINFOINDICATION_H_

#include "MessageType.hpp"
#include "AMF-UE-NGAP-ID.hpp"
#include "RAN-UE-NGAP-ID.hpp"
#include "UERadioCapability.hpp"
#include "UERadioCapabilityForPaging.hpp"

extern "C"{
	#include "Ngap_NGAP-PDU.h"
	#include "Ngap_ProtocolIE-Field.h"
	#include "Ngap_UERadioCapabilityInfoIndication.h"
}

namespace ngap{

	class UeRadioCapabilityInfoIndicationMsg{
	public:
		UeRadioCapabilityInfoIndicationMsg();
		virtual ~UeRadioCapabilityInfoIndicationMsg();
		
		void setMessageType();
		void setAmfUeNgapId(unsigned long id);//40 bits
		void setRanUeNgapId(uint32_t id);// 32 bits
		void setUERadioCapability (uint8_t *buf,size_t size);
		void setUERadioCapabilityForPaging(uint8_t * nr,size_t sizeofnr,uint8_t * eutra,size_t sizeofeutra);
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
		bool getAmfUeNgapId(unsigned long &id);
		bool getRanUeNgapId(uint32_t &id);
		bool getUERadioCapability(uint8_t *&buf,size_t &size);
		bool getUERadioCapabilityForPaging(uint8_t *&nr,size_t &sizeofnr, uint8_t *&eutra,size_t &sizeofeutra);
	private:
		Ngap_NGAP_PDU_t * ueRadioCapabilityInfoIndicationPdu;
		Ngap_UERadioCapabilityInfoIndication_t *ueRadioCapabilityInfoIndicationIEs;
		
		AMF_UE_NGAP_ID *amfUeNgapId;
		RAN_UE_NGAP_ID *ranUeNgapId;
		UERadioCapability *ueRadioCapability;
		UERadioCapabilityForPaging *ueRadioCapabilityForPaging;
	};

}
#endif


