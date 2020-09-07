#ifndef _PDUSESSIONRESOURCESETUPREQUESTTRANSFER_H_
#define _PDUSESSIONRESOURCESETUPREQUESTTRANSFER_H_

#include <vector>
#include "NgapIEsStruct.hpp"

#include "PduSessionAggregateMaximumBitRate.hpp"
#include "UPTransportLayerInformation.hpp"
#include "DataForwardingNotPossible.hpp"
#include "PDUSessionType.hpp"
#include "SecurityIndication.hpp"
#include "NetworkInstance.hpp"
#include "QosFlowSetupRequestList.hpp"


extern "C"{
	#include "Ngap_ProtocolIE-Field.h"
	#include "Ngap_PDUSessionResourceSetupRequestTransfer.h"
}

namespace ngap{

	class PduSessionResourceSetupRequestTransferIE{
		
	public:
		PduSessionResourceSetupRequestTransferIE();
		virtual ~PduSessionResourceSetupRequestTransferIE();		
		
		void setPduSessionAggregateMaximumBitRate(long bit_rate_downlink, long bit_rate_uplink);//O
		void setUL_NG_U_UP_TNL_Information(GtpTunnel_t upTnlInfo);//M
		//void setAdditionalUL_NG_U_UP_TNL_Information(std::vector<GtpTunnel>list);//O
		void setDataForwardingNotPossible();//O
		void setPduSessionType(e_Ngap_PDUSessionType type);//M
		void setSecurityIndication(e_Ngap_IntegrityProtectionIndication integrity_protection, e_Ngap_ConfidentialityProtectionIndication confidentiality_protection, e_Ngap_MaximumIntegrityProtectedDataRate maxIntProtDataRate);
		void setSecurityIndication(e_Ngap_IntegrityProtectionIndication integrity_protection, e_Ngap_ConfidentialityProtectionIndication confidentiality_protection);
		void setNetworkInstance(long value);
		void setQosFlowSetupRequestList(std::vector<QosFlowSetupReq_t>list);
		int  encode2buffer(uint8_t *buf, int buf_size);
		//Decapsulation
		bool decodefromIE(uint8_t *buf, int buf_size);
		bool getPduSessionAggregateMaximumBitRate(long &bit_rate_downlink, long &bit_rate_uplink);//O
		bool getUL_NG_U_UP_TNL_Information(GtpTunnel_t &upTnlInfo);//M
		//bool getAdditionalUL_NG_U_UP_TNL_Information(std::vector<GtpTunnel>&list);//O
		bool getDataForwardingNotPossible();//O
		bool getPduSessionType(long &type);//M
		bool getSecurityIndication(long &integrity_protection, long &confidentiality_protection, long &maxIntProtDataRate);
		bool getNetworkInstance(long &value);
		bool getQosFlowSetupRequestList(std::vector<QosFlowSetupReq_t> &list);

	private:
		Ngap_PDUSessionResourceSetupRequestTransfer_t *pduSessionResourceSetupRequestTransferIEs;

		PduSessionAggregateMaximumBitRate *pduSessionAggregateMaximumBitRateIE;
		UpTransportLayerInformation *upTransportLayerInformation;
		DataForwardingNotPossible *dataForwardingNotPossible;
		PDUSessionType *pduSessionType;
		SecurityIndication *securityIndication;
		NetworkInstance *networkInstance;
		QosFlowSetupRequestList *qosFlowSetupRequestList;

	};

}
#endif


