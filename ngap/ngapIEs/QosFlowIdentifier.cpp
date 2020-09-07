#include "QosFlowSetupRequestItem.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	QosFlowIdentifier::QosFlowIdentifier(){}  

	QosFlowIdentifier::~QosFlowIdentifier(){} 

	void QosFlowIdentifier::setQosFlowIdentifier(long value)
	{
		identifier = value;
	}
	bool QosFlowIdentifier::getQosFlowIdentifier(long &value)
	{
		value = identifier;
		
		return true;
	}
	
	bool QosFlowIdentifier::encode2QosFlowIdentifier(Ngap_QosFlowIdentifier_t *qosFlowIdentifier)
	{
		*qosFlowIdentifier = identifier;
	
		return true;
	}
	bool QosFlowIdentifier::decodefromQosFlowIdentifier(Ngap_QosFlowIdentifier_t *qosFlowIdentifier)
	{
		identifier = *qosFlowIdentifier;
		
		return true;
	}
}




