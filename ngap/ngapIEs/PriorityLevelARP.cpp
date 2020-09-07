#include "PriorityLevelARP.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	PriorityLevelARP::PriorityLevelARP(){}  

	PriorityLevelARP::~PriorityLevelARP(){} 

	void PriorityLevelARP::setPriorityLevelARP(long value)
	{
		prioritylevelarp = value;
	}
	bool PriorityLevelARP::getPriorityLevelARP(long &value)
	{
		value = prioritylevelarp;
		
		return true;
	}
	
	bool PriorityLevelARP::encode2PriorityLevelARP(Ngap_PriorityLevelARP_t *priorityLevelARP)
	{
		*priorityLevelARP = prioritylevelarp;
	
		return true;
	}
	bool PriorityLevelARP::decodefromPriorityLevelARP(Ngap_PriorityLevelARP_t *priorityLevelARP)
	{
		prioritylevelarp = *priorityLevelARP;
		
		return true;
	}
}
