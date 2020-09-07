#include "AMFRegionID.hpp"
#include "String2Value.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	AMFRegionID::AMFRegionID(){}  

	AMFRegionID::~AMFRegionID(){} 

	void AMFRegionID::setAMFRegionID(const std::string charid)
	{
		regionid = fromString<int>(charid);
	}
	void AMFRegionID::getAMFRegionID(std::string & charid)
	{
		charid = to_string(regionid);
	}
	
	bool AMFRegionID::encode2bitstring(Ngap_AMFRegionID_t &amfregionid)
	{
		amfregionid.size = 1;
		uint8_t *buffer = (uint8_t *)calloc(1,sizeof(uint8_t));
		if(!buffer) return false;
		*buffer = regionid;
		amfregionid.buf = buffer;
		amfregionid.bits_unused = 0;

		return true;
	}
	bool AMFRegionID::decodefrombitstring(Ngap_AMFRegionID_t &amfregionid)
	{
		if(!amfregionid.buf) return false;
		regionid = *amfregionid.buf;

		return true;
	}
}

