#include "MaximumDataBurstVolume.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	MaximumDataBurstVolume::MaximumDataBurstVolume(){}  

	MaximumDataBurstVolume::~MaximumDataBurstVolume(){} 

	void MaximumDataBurstVolume::setMaximumDataBurstVolume(long value)
	{
		maximumdataburstvolume = value;
	}
	bool MaximumDataBurstVolume::getMaximumDataBurstVolume(long &value)
	{
		value = maximumdataburstvolume;
		
		return true;
	}
	
	bool MaximumDataBurstVolume::encode2MaximumDataBurstVolume(Ngap_MaximumDataBurstVolume_t *maximumDataBurstVolume)
	{
		*maximumDataBurstVolume = maximumdataburstvolume;
	
		return true;
	}
	bool MaximumDataBurstVolume::decodefromMaximumDataBurstVolume(Ngap_MaximumDataBurstVolume_t *maximumDataBurstVolume)
	{
		maximumdataburstvolume = *maximumDataBurstVolume;
		
		return true;
	}
}

