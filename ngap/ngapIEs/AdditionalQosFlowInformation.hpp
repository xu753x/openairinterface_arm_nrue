#ifndef _ADDITIONALQOSFLOWINFORMATION_H_
#define _ADDITIONALQOSFLOWINFORMATION_H_

extern "C"{
  #include "Ngap_AdditionalQosFlowInformation.h"
}

namespace ngap{

	class AdditionalQosFlowInformation{
	public:
		AdditionalQosFlowInformation();
		virtual ~AdditionalQosFlowInformation();
		
		void setAdditionalQosFlowInformation(e_Ngap_AdditionalQosFlowInformation value);
		bool getAdditionalQosFlowInformation(e_Ngap_AdditionalQosFlowInformation &value);
		
		bool encode2AdditionalQosFlowInformation(Ngap_AdditionalQosFlowInformation_t *);
		bool decodefromAdditionalQosFlowInformation(Ngap_AdditionalQosFlowInformation_t *);
	private:
		long additional;
	};

}

#endif


