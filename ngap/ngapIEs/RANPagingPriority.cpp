#include "RANPagingPriority.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	RANPagingPriority::RANPagingPriority(){}  

	RANPagingPriority::~RANPagingPriority(){} 

	void RANPagingPriority::setRANPagingPriority(uint8_t m_ranPagingPriority)
	{
		ranPagingPriority = m_ranPagingPriority;
	}
	uint8_t RANPagingPriority::getRANPagingPriority()
	{
		return ranPagingPriority;
	}
	
	bool RANPagingPriority::encode2RANPagingPriority(Ngap_RANPagingPriority_t &ranpagingpriority)
	{
		ranpagingpriority = ranPagingPriority;
		return true;
	}
	bool RANPagingPriority::decodefromRANPagingPriority(Ngap_RANPagingPriority_t &ranpagingpriority)
	{
		ranPagingPriority = ranpagingpriority;
		return true;
	}
}



