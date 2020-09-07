#ifndef _PDUSESSIONRESOURCERELEASECOMMANDTRANSFER_H_
#define _PDUSESSIONRESOURCERELEASECOMMANDTRANSFER_H_

#include "NgapIEsStruct.hpp"

#include "Cause.hpp"


extern "C"{
	#include "Ngap_ProtocolIE-Field.h"
	#include "Ngap_PDUSessionResourceReleaseCommandTransfer.h"
}

namespace ngap{

	class PDUSessionResourceReleaseCommandTransfer {
		
	public:
		PDUSessionResourceReleaseCommandTransfer();
		virtual ~PDUSessionResourceReleaseCommandTransfer();
		
		void setCauseRadioNetwork(e_Ngap_CauseRadioNetwork cause_value);
		void setCauseTransport(e_Ngap_CauseTransport cause_value);
		void setCauseNas(e_Ngap_CauseNas cause_value);
		void setCauseProtocol(e_Ngap_CauseProtocol cause_value);
		void setCauseMisc(e_Ngap_CauseMisc cause_value);
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefromIE(uint8_t *buf, int buf_size);
		long getChoiceOfCause();
		long getCause();

	private:
		Ngap_PDUSessionResourceReleaseCommandTransfer_t *pduSessionResourceReleaseCommandTransferIEs;

		Cause *causeValue;

	};

}
#endif


