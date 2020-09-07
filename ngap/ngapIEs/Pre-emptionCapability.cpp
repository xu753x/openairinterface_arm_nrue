#include "Pre-emptionCapability.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	Pre_emptionCapability::Pre_emptionCapability(){}  

	Pre_emptionCapability::~Pre_emptionCapability(){} 

	void Pre_emptionCapability::setPre_emptionCapability(e_Ngap_Pre_emptionCapability value)
	{
		capability = value;
	}
	bool Pre_emptionCapability::getPre_emptionCapability(e_Ngap_Pre_emptionCapability &value)
	{
		if((capability==0) || (capability == 1))
			value = (e_Ngap_Pre_emptionCapability)capability;
		else
			return false;
		
		return true;
	}
	
	bool Pre_emptionCapability::encode2Pre_emptionCapability(Ngap_Pre_emptionCapability_t *pre_emptionCapability)
	{
		*pre_emptionCapability = capability;
	
		return true;
	}
	bool Pre_emptionCapability::decodefromPre_emptionCapability(Ngap_Pre_emptionCapability_t *pre_emptionCapability)
	{
		capability = *pre_emptionCapability;
		
		return true;
	}
}
