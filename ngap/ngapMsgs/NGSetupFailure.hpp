#ifndef _NGSETUPFAILURE_H_
#define _NGSETUPFAILURE_H_

#include "MessageType.hpp"
#include "Cause.hpp"
#include "TimeToWait.hpp"
//#include "CriticalityDiagnostics.hpp"

extern "C"{
	#include "Ngap_NGAP-PDU.h"
	#include "Ngap_NGSetupFailure.h"
	#include "Ngap_ProtocolIE-Field.h"
}

namespace ngap{

	class NGSetupFailureMsg{
	public:
		NGSetupFailureMsg();
		virtual ~NGSetupFailureMsg();

		//Erternal interfaces
		//Encapsulation
		void setMessageType();
		void setCauseRadioNetwork(e_Ngap_CauseRadioNetwork cause_value, e_Ngap_TimeToWait time_to_wait);
		void setCauseRadioNetwork(e_Ngap_CauseRadioNetwork cause_value);
		void setCauseTransport(e_Ngap_CauseTransport cause_value, e_Ngap_TimeToWait time_to_wait);
		void setCauseTransport(e_Ngap_CauseTransport cause_value);
		void setCauseNas(e_Ngap_CauseNas cause_value, e_Ngap_TimeToWait time_to_wait);
		void setCauseNas(e_Ngap_CauseNas cause_value);
		void setCauseProtocol(e_Ngap_CauseProtocol cause_value, e_Ngap_TimeToWait time_to_wait);
		void setCauseProtocol(e_Ngap_CauseProtocol cause_value);
		void setCauseMisc(e_Ngap_CauseMisc cause_value, e_Ngap_TimeToWait time_to_wait);
		void setCauseMisc(e_Ngap_CauseMisc cause_value);
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu);
		bool getCauseType(Ngap_Cause_PR &);
		bool getCauseRadioNetwork(e_Ngap_CauseRadioNetwork &);
		bool getCauseTransport(e_Ngap_CauseTransport &);
		bool getCauseNas(e_Ngap_CauseNas &);
		bool getCauseProtocol(e_Ngap_CauseProtocol &);
		bool getCauseMisc(e_Ngap_CauseMisc &);
		bool getTime2Wait(e_Ngap_TimeToWait &);
		
	private:
		Ngap_NGAP_PDU_t * ngSetupFailurePdu;
		Ngap_NGSetupFailure_t *ngSetupFailureIEs;

		Cause *cause;
		TimeToWait *timeToWait;
		//CriticalityDiagnostics *criticalityDiagnostics;

		void addCauseIE();
		void addTimeToWaitIE();
	};

}
#endif
