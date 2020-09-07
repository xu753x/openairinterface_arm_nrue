#ifndef _NGAPIESSTRUCT_H_
#define _NGAPIESSTRUCT_H_

#include <vector>
#include <string>

extern "C"{
  #include <OCTET_STRING.h>
  #include "Ngap_DelayCritical.h"
  #include "Ngap_Pre-emptionCapability.h"
  #include "Ngap_Pre-emptionVulnerability.h"
  #include "Ngap_NotificationControl.h"
  #include "Ngap_ReflectiveQosAttribute.h"
  #include "Ngap_AdditionalQosFlowInformation.h"
  #include "Ngap_AssociatedQosFlowItem.h"
}

namespace ngap{

	typedef struct SliceSupportItem_s{
		std::string sst;
		std::string sd;
	}SliceSupportItem_t;
	typedef struct PlmnSliceSupport_s{
		std::string mcc;
		std::string mnc;
		std::vector<SliceSupportItem_t> slice_list;
	}PlmnSliceSupport_t;
	
	typedef struct SupportedItem_s{  
		uint32_t tac;  
		std::vector<PlmnSliceSupport_t> b_plmn_list;
	}SupportedItem_t;

	typedef struct GuamiItem_s{
		std::string mcc;
		std::string mnc;
		std::string regionID;
		std::string AmfSetID;
		std::string AmfPointer;
		std::string backupAMFName;//optional
	}GuamiItem_t;

	typedef struct NrCgi_s{
		std::string mcc;
		std::string mnc;
		unsigned long nrCellID;
	}NrCgi_t;

	typedef struct Tai_s{
		std::string mcc;
		std::string mnc;
		uint32_t tac:24;
	}Tai_t;
		
	typedef struct Guami_s{
		std::string mcc;
		std::string mnc;
		std::string regionID;
		std::string AmfSetID;
		std::string AmfPointer;
	}Guami_t;
	
	typedef struct{
		std::string sst;
		std::string sd;//可选
	}S_Nssai;
	
	typedef struct{
		uint8_t pduSessionId;
		uint8_t *pduSessionNAS_PDU;
		size_t sizeofpduSessionNAS_PDU;
		S_Nssai s_nssai;
		OCTET_STRING_t pduSessionResourceSetupRequestTransfer;
	}PDUSessionResourceSetupRequestItem_t;
	
	typedef struct{
		std::string ip_address;
		uint32_t gtp_teid;
	}GtpTunnel_t;
	
	typedef struct{
		long _5QI;
		long *priorityLevelQos;
		long *averagingWindow;
		long *maximumDataBurstVolume;
	}NonDynamic5QI_t;
	
	typedef struct{
		long pERScalar;
		long pERExponent;
	}PacketErrorRate_t;
	
	typedef struct{
		long priorityLevelQos;
		long packetDelayBudget;
		PacketErrorRate_t packetErrorRate;
		long *_5QI;
		e_Ngap_DelayCritical *delayCritical;
		long *averagingWindow;
		long *maximumDataBurstVolume;
	}Dynamic5QI_t;
	
	typedef struct{
		NonDynamic5QI_t *nonDy;
		Dynamic5QI_t	*dy;
	}QosCharacteristics_t;
	
	typedef struct{
		long priorityLevelARP;
		e_Ngap_Pre_emptionCapability pre_emptionCapability;
		e_Ngap_Pre_emptionVulnerability pre_emptionVulnerability;
	}AllocationAndRetentionPriority_t;
	
	typedef struct{
		long maximumFlowBitRateDL;
		long maximumFlowBitRateUL;
		long guaranteedFlowBitRateDL;
		long guaranteedFlowBitRateUL;
		e_Ngap_NotificationControl *notificationControl;
		long *maximumPacketLossRateDL;
		long *maximumPacketLossRateUL;
	}GBR_QosInformation_t;
	
	typedef struct{
		QosCharacteristics_t qosc;
		AllocationAndRetentionPriority_t arp;
		GBR_QosInformation_t *gbr_qos_info;
		e_Ngap_ReflectiveQosAttribute *reflectiveQosAttribute;
		e_Ngap_AdditionalQosFlowInformation *additionalQosFlowInformation;
	}QosFlowLevelQosParameters_t;
	
	typedef struct{
		long qos_flow_id;
		QosFlowLevelQosParameters_t qflqp;
	}QosFlowSetupReq_t;

	typedef struct{
		long qosFlowIdentifier;
		e_Ngap_AssociatedQosFlowItem__qosFlowMappingIndication *qosFlowMappingIndication;
	}AssociatedQosFlow_t;

	
	typedef struct{
	uint8_t pduSessionId;
	OCTET_STRING_t pduSessionResourceSetupResponseTransfer;
	}PDUSessionResourceSetupResponseItem_t;
	
	typedef struct{
	uint8_t pduSessionId;
	OCTET_STRING_t pduSessionResourceSetupUnsuccessfulTransfer;
	}PDUSessionResourceFailedToSetupItem_t;
       
        typedef struct {
	uint8_t pduSessionId;
	OCTET_STRING_t pduSessionResourceReleaseCommandTransfer;
	}PDUSessionResourceReleaseCommandItem_t;

}

#endif
