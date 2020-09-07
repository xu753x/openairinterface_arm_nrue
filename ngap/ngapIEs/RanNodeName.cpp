#include "RanNodeName.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	RanNodeName::RanNodeName()
	{
		ranNodeName = NULL;
	}
	RanNodeName::~RanNodeName(){}

	void RanNodeName::setValue(const std::string ranName)
	{
		ranNodeName = (char *)ranName.c_str();
	}
	bool RanNodeName::getValue(std::string &ranName)
	{
		if(!ranNodeName) return false;
		ranName = ranNodeName;
		return true;
	}

	bool RanNodeName::encode2RanNodeName(Ngap_RANNodeName_t * rannodename)
	{
		int ret = OCTET_STRING_fromBuf (rannodename, ranNodeName, strlen(ranNodeName));

		if(ret == 0)
			return true;
		else
			return false;
	}
	bool RanNodeName::decodefromRanNodeName(Ngap_RANNodeName_t * rannodename)
	{
		if(!rannodename->buf) return false;
		ranNodeName = (char *)rannodename->buf;
		return true;
	}

}
