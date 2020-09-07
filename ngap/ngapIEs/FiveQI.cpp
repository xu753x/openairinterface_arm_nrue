#include "FiveQI.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	FiveQI::FiveQI(){}  

	FiveQI::~FiveQI(){} 

	void FiveQI::setFiveQI(long value)
	{
		fiveqi = value;
	}
	bool FiveQI::getFiveQI(long &value)
	{
		value = fiveqi;
		
		return true;
	}
	
	bool FiveQI::encode2FiveQI(Ngap_FiveQI_t *fiveQI)
	{
		*fiveQI = fiveqi;
	
		return true;
	}
	bool FiveQI::decodefromFiveQI(Ngap_FiveQI_t *fiveQI)
	{
		fiveqi = *fiveQI;
		
		return true;
	}
}





