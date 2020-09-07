#include "QosFlowSetupRequestItem.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	QosFlowSetupRequestItem::QosFlowSetupRequestItem()
	{
		qosFlowIdentifier = NULL;
		qosFlowLevelQosParameters = NULL;
	}  

	QosFlowSetupRequestItem::~QosFlowSetupRequestItem(){} 

	void QosFlowSetupRequestItem::setQosFlowSetupRequestItem(QosFlowIdentifier *m_qosFlowIdentifier,QosFlowLevelQosParameters *m_qosFlowLevelQosParameters)
	{
		qosFlowIdentifier = m_qosFlowIdentifier;
		qosFlowLevelQosParameters = m_qosFlowLevelQosParameters;
	}
	bool QosFlowSetupRequestItem::getQosFlowSetupRequestItem(QosFlowIdentifier *&m_qosFlowIdentifier,QosFlowLevelQosParameters *&m_qosFlowLevelQosParameters)
	{
		m_qosFlowIdentifier = qosFlowIdentifier;
		m_qosFlowLevelQosParameters = qosFlowLevelQosParameters;
	
		return true;
	}
	
	bool QosFlowSetupRequestItem::encode2QosFlowSetupRequestItem(Ngap_QosFlowSetupRequestItem_t *qosFlowSetupRequestItem)
	{
		if(!qosFlowIdentifier->encode2QosFlowIdentifier(&qosFlowSetupRequestItem->qosFlowIdentifier)) return false;
		if(!qosFlowLevelQosParameters->encode2QosFlowLevelQosParameters(&qosFlowSetupRequestItem->qosFlowLevelQosParameters)) return false;
		
		return true;
	}
	bool QosFlowSetupRequestItem::decodefromQosFlowSetupRequestItem(Ngap_QosFlowSetupRequestItem_t *qosFlowSetupRequestItem)
	{
		qosFlowIdentifier = new QosFlowIdentifier();
		qosFlowLevelQosParameters = new QosFlowLevelQosParameters();

		if(!qosFlowIdentifier->decodefromQosFlowIdentifier(&qosFlowSetupRequestItem->qosFlowIdentifier)) return false;
		if(!qosFlowLevelQosParameters->decodefromQosFlowLevelQosParameters(&qosFlowSetupRequestItem->qosFlowLevelQosParameters)) return false;
		
		return true;
	}
}



