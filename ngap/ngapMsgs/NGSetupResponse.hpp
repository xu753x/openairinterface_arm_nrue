#ifndef _NGSETUPRESPONSE_H_
#define _NGSETUPRESPONSE_H_

#include "NgapIEsStruct.hpp"

#include "MessageType.hpp"
#include "AMFName.hpp"
#include "ServedGUAMIList.hpp"
#include "RelativeAMFCapacity.hpp"
#include "PLMNSupportList.hpp"
//#include "CriticalityDiagnostics.hpp"

extern "C"{
	#include "Ngap_NGAP-PDU.h"
	#include "Ngap_NGSetupResponse.h"
	#include "Ngap_ProtocolIE-Field.h"
}

namespace ngap{

	class NGSetupResponseMsg{
	public:
		NGSetupResponseMsg();
		virtual ~NGSetupResponseMsg();

		//Erternal interfaces
		//Encapsulation
		void setMessageType(); //Initialize the PDU and populate the MessageType;
		void setAMFName(const std::string name);//ÃÓ≥‰IE
		void setGUAMIList(std::vector<struct GuamiItem_s> list);
		void setRelativeAmfCapacity(long capacity);
		void setPlmnSupportList(std::vector<PlmnSliceSupport_t> list);
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
		bool getAMFName(std::string &name);
		bool getGUAMIList(std::vector<struct GuamiItem_s> &list);
		long getRelativeAmfCapacity();
		bool getPlmnSupportList(std::vector<PlmnSliceSupport_t> &list);

	private:
		Ngap_NGAP_PDU_t * ngSetupResponsePdu;
		Ngap_NGSetupResponse_t *ngSetupResponsIEs;

	/***************** for decoding ****************/

		AmfName *amfName;
		ServedGUAMIList *servedGUAMIList;
		RelativeAMFCapacity *relativeAmfCapacity;
		PLMNSupportList *plmnSupportList;
		//CriticalityDiagnostics *criticalityDiagnostics;
	};

}
#endif

