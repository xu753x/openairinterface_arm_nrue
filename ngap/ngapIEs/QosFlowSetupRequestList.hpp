#ifndef _QOSFLOWSETUPREQUESTLIST_H_
#define _QOSFLOWSETUPREQUESTLIST_H_

#include "QosFlowSetupRequestItem.hpp"

extern "C"{
  #include "Ngap_QosFlowSetupRequestList.h"
}

namespace ngap{

	class QosFlowSetupRequestList{
	public:
		QosFlowSetupRequestList();
		virtual ~QosFlowSetupRequestList();
		
		void setQosFlowSetupRequestList(QosFlowSetupRequestItem * m_items, int m_numofitems);
		bool getQosFlowSetupRequestList(QosFlowSetupRequestItem * & m_items, int & m_numofitems);
		bool encode2QosFlowSetupRequestList(Ngap_QosFlowSetupRequestList_t *qosFlowSetupRequestList);
		bool decodefromQosFlowSetupRequestList(Ngap_QosFlowSetupRequestList_t *qosFlowSetupRequestList);
	private:
		QosFlowSetupRequestItem * items;
		int numofitems;
	};
}
#endif
