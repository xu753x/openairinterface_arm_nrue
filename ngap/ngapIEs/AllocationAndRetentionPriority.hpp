#ifndef _ALLOCATIONANDRETENTIONPRIORITY_H_
#define _ALLOCATIONANDRETENTIONPRIORITY_H_

#include "PriorityLevelARP.hpp"
#include "Pre-emptionCapability.hpp"
#include "Pre-emptionVulnerability.hpp"

extern "C"{
  #include "Ngap_AllocationAndRetentionPriority.h"
}

namespace ngap{

	class AllocationAndRetentionPriority{
	public:
		AllocationAndRetentionPriority();
		virtual ~AllocationAndRetentionPriority();
		
		void setAllocationAndRetentionPriority(PriorityLevelARP *m_priorityLevelARP,Pre_emptionCapability *m_pre_emptionCapability,Pre_emptionVulnerability *m_pre_emptionVulnerability);
		bool getAllocationAndRetentionPriority(PriorityLevelARP *&m_priorityLevelARP,Pre_emptionCapability *&m_pre_emptionCapability,Pre_emptionVulnerability *&m_pre_emptionVulnerability);
		
		bool encode2AllocationAndRetentionPriority(Ngap_AllocationAndRetentionPriority_t *);
		bool decodefromAllocationAndRetentionPriority(Ngap_AllocationAndRetentionPriority_t *);
	private:
		PriorityLevelARP *priorityLevelARP;
		Pre_emptionCapability *pre_emptionCapability;
		Pre_emptionVulnerability *pre_emptionVulnerability;
	};
}

#endif
