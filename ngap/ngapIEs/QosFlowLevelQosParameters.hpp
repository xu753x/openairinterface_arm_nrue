#ifndef _QOSFLOWLEVELQOSPARAMETERS_H_
#define _QOSFLOWLEVELQOSPARAMETERS_H_

#include"QosCharacteristics.hpp"
#include"AllocationAndRetentionPriority.hpp"
#include"GBR-QosInformation.hpp"
#include"ReflectiveQosAttribute.hpp"
#include"AdditionalQosFlowInformation.hpp"

extern "C"{
  #include "Ngap_QosFlowLevelQosParameters.h"
}

namespace ngap{

	class QosFlowLevelQosParameters{
	public:
		QosFlowLevelQosParameters();
		virtual ~QosFlowLevelQosParameters();
		
		void setQosFlowLevelQosParameters(QosCharacteristics *m_qosCharacteristics,AllocationAndRetentionPriority *m_allocationAndRetentionPriority,GBR_QosInformation *m_gBR_QosInformation,ReflectiveQosAttribute *m_reflectiveQosAttribute,AdditionalQosFlowInformation *m_additionalQosFlowInformation);
		bool getQosFlowLevelQosParameters(QosCharacteristics *&m_qosCharacteristics,AllocationAndRetentionPriority *&m_allocationAndRetentionPriority,GBR_QosInformation *&m_gBR_QosInformation,ReflectiveQosAttribute *&m_reflectiveQosAttribute,AdditionalQosFlowInformation *&m_additionalQosFlowInformation);
		
		bool encode2QosFlowLevelQosParameters(Ngap_QosFlowLevelQosParameters_t *);
		bool decodefromQosFlowLevelQosParameters(Ngap_QosFlowLevelQosParameters_t *);
	private:
		QosCharacteristics *qosCharacteristics;
		AllocationAndRetentionPriority *allocationAndRetentionPriority;
		GBR_QosInformation *gBR_QosInformation;	/* OPTIONAL */
		ReflectiveQosAttribute *reflectiveQosAttribute;	/* OPTIONAL */
		AdditionalQosFlowInformation *additionalQosFlowInformation;	/* OPTIONAL */
	};

}

#endif

