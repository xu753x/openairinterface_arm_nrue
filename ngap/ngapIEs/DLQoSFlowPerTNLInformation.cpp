#include "DLQoSFlowPerTNLInformation.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	DLQoSFlowPerTNLInformation::DLQoSFlowPerTNLInformation()
	{
		uPTransportLayerInformation = NULL;
		associatedQosFlowList = NULL;
	}  

	DLQoSFlowPerTNLInformation::~DLQoSFlowPerTNLInformation(){} 
	
	
	void DLQoSFlowPerTNLInformation::setDLQoSFlowPerTNLInformation(UpTransportLayerInformation *m_uPTransportLayerInformation, AssociatedQosFlowList *m_associatedQosFlowList)
	{
		uPTransportLayerInformation = m_uPTransportLayerInformation;
		associatedQosFlowList = m_associatedQosFlowList;
	}
	bool DLQoSFlowPerTNLInformation::getDLQoSFlowPerTNLInformation(UpTransportLayerInformation *&m_uPTransportLayerInformation, AssociatedQosFlowList *&m_associatedQosFlowList)
	{
		m_uPTransportLayerInformation = uPTransportLayerInformation;
		m_associatedQosFlowList = associatedQosFlowList;
		
		return true;
	}
	
	bool DLQoSFlowPerTNLInformation::encode2DLQoSFlowPerTNLInformation(Ngap_QosFlowPerTNLInformation_t *qosFlowPerTNLInformation)
	{
		if(!uPTransportLayerInformation->encode2UpTransportLayerInformation(qosFlowPerTNLInformation->uPTransportLayerInformation)) return false;
		if(!associatedQosFlowList->encode2AssociatedQosFlowList(qosFlowPerTNLInformation->associatedQosFlowList)) return false;

		return true;
	}
	bool DLQoSFlowPerTNLInformation::decodefromDLQoSFlowPerTNLInformation(Ngap_QosFlowPerTNLInformation_t *qosFlowPerTNLInformation)
	{
		uPTransportLayerInformation = new UpTransportLayerInformation();
		associatedQosFlowList = new AssociatedQosFlowList();
		if(!uPTransportLayerInformation->decodefromUpTransportLayerInformation(qosFlowPerTNLInformation->uPTransportLayerInformation)) return false;
		if(!associatedQosFlowList->decodefromAssociatedQosFlowList(qosFlowPerTNLInformation->associatedQosFlowList)) return false;
		
		return true;
	}

}




