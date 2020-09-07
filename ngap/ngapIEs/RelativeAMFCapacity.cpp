#include "RelativeAMFCapacity.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	RelativeAMFCapacity::RelativeAMFCapacity(){}
	RelativeAMFCapacity::~RelativeAMFCapacity(){}
	
	void RelativeAMFCapacity::setValue(long m_amfcapacity)
	{
		amfcapacity = m_amfcapacity;
		if(m_amfcapacity > 255)
			cout<<"[Warning] RelativeAMFCapacity > 255 !"<<endl;
	}
	long RelativeAMFCapacity::getValue()
	{
		return amfcapacity;
	}

	bool RelativeAMFCapacity::encode2RelativeAMFCapacity(Ngap_RelativeAMFCapacity_t * amfCapacityIe)
	{   
		*amfCapacityIe = amfcapacity;
		
		return true;
	}
	bool RelativeAMFCapacity::decodefromRelativeAMFCapacity(Ngap_RelativeAMFCapacity_t * pdu)
	{
		amfcapacity = *pdu;
		
		return true;
	}

}


