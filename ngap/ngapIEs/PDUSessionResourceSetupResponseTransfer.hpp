#ifndef _PDUSESSIONRESOURCESETUPRESPONSETRANSFER_H_
#define _PDUSESSIONRESOURCESETUPRESPONSETRANSFER_H_

#include "NgapIEsStruct.hpp"

#include "DLQoSFlowPerTNLInformation.hpp"
#include "SecurityResult.hpp"

extern "C"{
	#include "Ngap_ProtocolIE-Field.h"
	#include "Ngap_PDUSessionResourceSetupResponseTransfer.h"
}

namespace ngap{

	class PduSessionResourceSetupResponseTransferIE{
		
	public:
		PduSessionResourceSetupResponseTransferIE();
		virtual ~PduSessionResourceSetupResponseTransferIE();		
		
		void setDLQoSFlowPerTNLInformation(GtpTunnel_t uptlinfo, std::vector<AssociatedQosFlow_t>list);
		void setAdditionalDLQoSFlowPerTNLInformation(GtpTunnel_t uptlinfo, std::vector<AssociatedQosFlow_t>list);
		void setSecurityResult(e_Ngap_IntegrityProtectionResult integrityProtectionResult,e_Ngap_ConfidentialityProtectionResult confidentialityProtectionResult);
		//caus待续

		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefromIE(uint8_t *buf, int buf_size);
		bool getDLQoSFlowPerTNLInformation(GtpTunnel_t &uptlinfo, std::vector<AssociatedQosFlow_t>&list);
		bool getAdditionalDLQoSFlowPerTNLInformation(GtpTunnel_t &uptlinfo, std::vector<AssociatedQosFlow_t>&list);
		bool getSecurityResult(long &integrityProtectionResult,long &confidentialityProtectionResult);
		//caus待续
		
	private:
		Ngap_PDUSessionResourceSetupResponseTransfer_t *pduSessionResourceSetupResponseTransferIEs;

		DLQoSFlowPerTNLInformation *dlQoSFlowPerTNLInformation;
		DLQoSFlowPerTNLInformation *additionalDLQoSFlowPerTNLInformation;
		SecurityResult *securityResult;
	};

}
#endif



