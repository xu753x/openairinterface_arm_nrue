#include "DelayCritical.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	DelayCritical::DelayCritical(){}  

	DelayCritical::~DelayCritical(){} 

	void DelayCritical::setDelayCritical(e_Ngap_DelayCritical value)
	{
		delaycritical = value;
	}
	bool DelayCritical::getDelayCritical(e_Ngap_DelayCritical &value)
	{
		if((delaycritical>=0) && (delaycritical<=1))
			value = (e_Ngap_DelayCritical)delaycritical;
		else
			false;
		
		return true;
	}
	
	bool DelayCritical::encode2DelayCritical(Ngap_DelayCritical_t *delayCritical)
	{
		*delayCritical = delaycritical;
	
		return true;
	}
	bool DelayCritical::decodefromDelayCritical(Ngap_DelayCritical_t *delayCritical)
	{
		delaycritical = *delayCritical;
		
		return true;
	}
}
