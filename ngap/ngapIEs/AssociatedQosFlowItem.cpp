#include "AssociatedQosFlowItem.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	AssociatedQosFlowItem::AssociatedQosFlowItem()
	{
		qosFlowIdentifier = NULL;
		qosFlowMappingIndication = -1;
	}  

	AssociatedQosFlowItem::~AssociatedQosFlowItem(){} 
	
	
	void AssociatedQosFlowItem::setAssociatedQosFlowItem(e_Ngap_AssociatedQosFlowItem__qosFlowMappingIndication m_qosFlowMappingIndication,QosFlowIdentifier *m_qosFlowIdentifier)
	{
		qosFlowMappingIndication = m_qosFlowMappingIndication;
		qosFlowIdentifier = m_qosFlowIdentifier;
	}
	void AssociatedQosFlowItem::setAssociatedQosFlowItem(QosFlowIdentifier *m_qosFlowIdentifier)
	{
		qosFlowIdentifier = m_qosFlowIdentifier;
	}
	bool AssociatedQosFlowItem::getAssociatedQosFlowItem(long &m_qosFlowMappingIndication,QosFlowIdentifier *&m_qosFlowIdentifier)
	{
		m_qosFlowMappingIndication = qosFlowMappingIndication;
		m_qosFlowIdentifier = qosFlowIdentifier;
		
		return true;
	}
	
	bool AssociatedQosFlowItem::encode2AssociatedQosFlowItem(Ngap_AssociatedQosFlowItem_t *associatedQosFlowItem)
	{
		if(qosFlowMappingIndication >= 0)
		{
			associatedQosFlowItem->qosFlowMappingIndication = (long *)calloc(1,sizeof(long));
			*associatedQosFlowItem->qosFlowMappingIndication = qosFlowMappingIndication;
		}
		if(!qosFlowIdentifier->encode2QosFlowIdentifier(&associatedQosFlowItem->qosFlowIdentifier)) return false;

		return true;
	}
	bool AssociatedQosFlowItem::decodefromAssociatedQosFlowItem(Ngap_AssociatedQosFlowItem_t *associatedQosFlowItem)
	{
		qosFlowIdentifier = new QosFlowIdentifier();
		if(!qosFlowIdentifier->decodefromQosFlowIdentifier(&associatedQosFlowItem->qosFlowIdentifier)) return false;
		
		if(associatedQosFlowItem->qosFlowMappingIndication)
		{
			qosFlowMappingIndication = *associatedQosFlowItem->qosFlowMappingIndication;
		}

		return true;
	}

}



