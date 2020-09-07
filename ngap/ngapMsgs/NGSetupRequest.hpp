#ifndef _NGSETUPREQUEST_H_
#define _NGSETUPREQUEST_H_

#include "NgapIEsStruct.hpp"

#include "MessageType.hpp"
#include "GlobalRanNodeId.hpp"
#include "RanNodeName.hpp"
#include "DefaultPagingDRX.hpp"
#include "SupportedTAList.hpp"

extern "C"{
  #include "Ngap_NGAP-PDU.h"
  #include "Ngap_ProtocolIE-Field.h"
  #include "Ngap_NGSetupRequest.h"
}

namespace ngap{

	class NGSetupRequestMsg{
	public:	
		NGSetupRequestMsg();
		virtual ~NGSetupRequestMsg();

		//Erternal interfaces
		//Encapsulation
		void setMessageType(); //Initialize the PDU and populate the MessageType;
		void setGlobalRanNodeID(const std::string mcc, const std::string mnc, Ngap_GlobalRANNodeID_PR ranNodeType, uint32_t ranNodeId);
		void setRanNodeName(const std::string ranNodeName);
		void setSupportedTAList(const std::vector<struct SupportedItem_s> list);
		void setDefaultPagingDRX(e_Ngap_PagingDRX value);
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
		bool getGlobalGnbID(uint32_t &gnbId,std::string &mcc,std::string &mnc);
		bool getRanNodeName(std::string &name);
		bool getSupportedTAList(std::vector<struct SupportedItem_s> &list);
		int  getDefaultPagingDRX();
		
	private: 
		Ngap_NGAP_PDU_t * ngSetupRequestPdu;
		Ngap_NGSetupRequest_t *ngSetupRequestIEs;

		/***************** for decoding ****************/
		GlobalRanNodeId * globalRanNodeId;
		RanNodeName * ranNodeName;
		SupportedTAList * supportedTAList;
		DefaultPagingDRX * defaultPagingDrx;
	};

}

#endif
