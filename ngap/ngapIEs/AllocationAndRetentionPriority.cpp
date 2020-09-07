#include "AllocationAndRetentionPriority.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	AllocationAndRetentionPriority::AllocationAndRetentionPriority()
	{
		priorityLevelARP = NULL;
		pre_emptionCapability = NULL;
		pre_emptionVulnerability = NULL;
	}  

	AllocationAndRetentionPriority::~AllocationAndRetentionPriority(){} 

	void AllocationAndRetentionPriority::setAllocationAndRetentionPriority(PriorityLevelARP *m_priorityLevelARP,Pre_emptionCapability *m_pre_emptionCapability,Pre_emptionVulnerability *m_pre_emptionVulnerability)
	{
		priorityLevelARP = m_priorityLevelARP;
		pre_emptionCapability = m_pre_emptionCapability;
		pre_emptionVulnerability = m_pre_emptionVulnerability;
	}
	bool AllocationAndRetentionPriority::getAllocationAndRetentionPriority(PriorityLevelARP *&m_priorityLevelARP,Pre_emptionCapability *&m_pre_emptionCapability,Pre_emptionVulnerability *&m_pre_emptionVulnerability)
	{
		if(!priorityLevelARP) return false;
		m_priorityLevelARP = priorityLevelARP;
		if(!pre_emptionCapability) return false;
		m_pre_emptionCapability = pre_emptionCapability;
		if(!pre_emptionVulnerability) return false;
		m_pre_emptionVulnerability = pre_emptionVulnerability;
		
		return true;
	}
	bool AllocationAndRetentionPriority::encode2AllocationAndRetentionPriority(Ngap_AllocationAndRetentionPriority_t *allocationAndRetentionPriority)
	{
		if(!priorityLevelARP->encode2PriorityLevelARP(&allocationAndRetentionPriority->priorityLevelARP)) return false;
		if(!pre_emptionCapability->encode2Pre_emptionCapability(&allocationAndRetentionPriority->pre_emptionCapability)) return false;
		if(!pre_emptionVulnerability->encode2Pre_emptionVulnerability(&allocationAndRetentionPriority->pre_emptionVulnerability)) return false;
		
		return true;
	}
	bool AllocationAndRetentionPriority::decodefromAllocationAndRetentionPriority(Ngap_AllocationAndRetentionPriority_t *allocationAndRetentionPriority)
	{
		priorityLevelARP = new PriorityLevelARP();
		pre_emptionCapability = new Pre_emptionCapability();
		pre_emptionVulnerability = new Pre_emptionVulnerability();
		if(!priorityLevelARP->decodefromPriorityLevelARP(&allocationAndRetentionPriority->priorityLevelARP)) return false;
		if(!pre_emptionCapability->decodefromPre_emptionCapability(&allocationAndRetentionPriority->pre_emptionCapability)) return false;
		if(!pre_emptionVulnerability->decodefromPre_emptionVulnerability(&allocationAndRetentionPriority->pre_emptionVulnerability)) return false;
		
		return true;
	}
}






