#ifndef _ASSOCIATEDQOSFLOWITEM_H_
#define _ASSOCIATEDQOSFLOWITEM_H_

#include "QosFlowIdentifier.hpp"

extern "C"{
  #include "Ngap_AssociatedQosFlowItem.h"
}

namespace ngap{

	class AssociatedQosFlowItem{
	public:
		AssociatedQosFlowItem();
		virtual ~AssociatedQosFlowItem();
		
		void setAssociatedQosFlowItem(e_Ngap_AssociatedQosFlowItem__qosFlowMappingIndication m_qosFlowMappingIndication,QosFlowIdentifier *m_qosFlowIdentifier);
		void setAssociatedQosFlowItem(QosFlowIdentifier *m_qosFlowIdentifier);
		bool getAssociatedQosFlowItem(long &m_qosFlowMappingIndication,QosFlowIdentifier *&m_qosFlowIdentifier);
		
		bool encode2AssociatedQosFlowItem(Ngap_AssociatedQosFlowItem_t *associatedQosFlowItem);
		bool decodefromAssociatedQosFlowItem(Ngap_AssociatedQosFlowItem_t *associatedQosFlowItem);
	private:
		QosFlowIdentifier *qosFlowIdentifier;
		long qosFlowMappingIndication;
	};

}

#endif

