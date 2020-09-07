#ifndef _QOSFLOWSETUPREQUESTITEM_H_
#define _QOSFLOWSETUPREQUESTITEM_H_

#include "QosFlowIdentifier.hpp"
#include "QosFlowLevelQosParameters.hpp"

extern "C"{
  #include "Ngap_QosFlowSetupRequestItem.h"
}

namespace ngap{

	class QosFlowSetupRequestItem{
	public:
		QosFlowSetupRequestItem();
		virtual ~QosFlowSetupRequestItem();
		
		void setQosFlowSetupRequestItem(QosFlowIdentifier *m_qosFlowIdentifier,QosFlowLevelQosParameters *m_qosFlowLevelQosParameters);
		bool getQosFlowSetupRequestItem(QosFlowIdentifier *&m_qosFlowIdentifier,QosFlowLevelQosParameters *&m_qosFlowLevelQosParameters);
		
		bool encode2QosFlowSetupRequestItem(Ngap_QosFlowSetupRequestItem_t *);
		bool decodefromQosFlowSetupRequestItem(Ngap_QosFlowSetupRequestItem_t *);
	private:
		QosFlowIdentifier *qosFlowIdentifier;
		QosFlowLevelQosParameters *qosFlowLevelQosParameters;
	};

}

#endif
