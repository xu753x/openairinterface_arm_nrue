#include "PriorityLevelQos.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	PriorityLevelQos::PriorityLevelQos(){}  

	PriorityLevelQos::~PriorityLevelQos(){} 

	void PriorityLevelQos::setPriorityLevelQos(long value)
	{
		prioritylevelqos = value;
	}
	bool PriorityLevelQos::getPriorityLevelQos(long &value)
	{
		value = prioritylevelqos;
		
		return true;
	}
	
	bool PriorityLevelQos::encode2PriorityLevelQos(Ngap_PriorityLevelQos_t *priorityLevelQos)
	{
		*priorityLevelQos = prioritylevelqos;
	
		return true;
	}
	bool PriorityLevelQos::decodefromPriorityLevelQos(Ngap_PriorityLevelQos_t *priorityLevelQos)
	{
		prioritylevelqos = *priorityLevelQos;
		
		return true;
	}
}






