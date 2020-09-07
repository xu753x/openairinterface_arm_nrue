#ifndef _DLQOSFLOWPERTNLINFORMATION_H_
#define _DLQOSFLOWPERTNLINFORMATION_H_

#include "UPTransportLayerInformation.hpp"
#include "AssociatedQosFlowList.hpp"

extern "C"{
  #include "Ngap_QosFlowPerTNLInformation.h"
}

namespace ngap{

	class DLQoSFlowPerTNLInformation{
	public:
		DLQoSFlowPerTNLInformation();
		virtual ~DLQoSFlowPerTNLInformation();
		
		void setDLQoSFlowPerTNLInformation(UpTransportLayerInformation *m_uPTransportLayerInformation, AssociatedQosFlowList *m_associatedQosFlowList);
		bool getDLQoSFlowPerTNLInformation(UpTransportLayerInformation *&m_uPTransportLayerInformation, AssociatedQosFlowList *&m_associatedQosFlowList);
		
		bool encode2DLQoSFlowPerTNLInformation(Ngap_QosFlowPerTNLInformation_t *qosFlowPerTNLInformation);
		bool decodefromDLQoSFlowPerTNLInformation(Ngap_QosFlowPerTNLInformation_t *qosFlowPerTNLInformation);
	private:
		UpTransportLayerInformation *uPTransportLayerInformation;
		AssociatedQosFlowList *associatedQosFlowList;
	};

}

#endif


