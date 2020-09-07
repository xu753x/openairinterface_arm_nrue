#include "AdditionalQosFlowInformation.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	AdditionalQosFlowInformation::AdditionalQosFlowInformation(){}  

	AdditionalQosFlowInformation::~AdditionalQosFlowInformation(){} 

	void AdditionalQosFlowInformation::setAdditionalQosFlowInformation(e_Ngap_AdditionalQosFlowInformation value)
	{
		additional = value;
	}
	bool AdditionalQosFlowInformation::getAdditionalQosFlowInformation(e_Ngap_AdditionalQosFlowInformation &value)
	{
		if(additional == Ngap_AdditionalQosFlowInformation_more_likely)
			value = (e_Ngap_AdditionalQosFlowInformation)additional;
		else
			return false;
		
		return true;
	}
	
	bool AdditionalQosFlowInformation::encode2AdditionalQosFlowInformation(Ngap_AdditionalQosFlowInformation_t *additionalQosFlowInfo)
	{
		*additionalQosFlowInfo = additional;
	
		return true;
	}
	bool AdditionalQosFlowInformation::decodefromAdditionalQosFlowInformation(Ngap_AdditionalQosFlowInformation_t *additionalQosFlowInfo)
	{
		additional = *additionalQosFlowInfo;
		
		return true;
	}
}
