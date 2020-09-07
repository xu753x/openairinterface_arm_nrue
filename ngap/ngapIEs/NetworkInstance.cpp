#include "NetworkInstance.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	NetworkInstance::NetworkInstance(){}  

	NetworkInstance::~NetworkInstance(){} 
	
	
	void NetworkInstance::setNetworkInstance(long m_networkinstance)
	{
		networkinstance = m_networkinstance;
	}
	bool NetworkInstance::getNetworkInstance(long &m_networkinstance)
	{
		m_networkinstance = networkinstance;
		
		return true;
	}
	
	bool NetworkInstance::encode2NetworkInstance(Ngap_NetworkInstance_t &value)
	{
		value = networkinstance;

		return true;
	}
	bool NetworkInstance::decodefromNetworkInstance(Ngap_NetworkInstance_t value)
	{
		networkinstance = value;
		
		return true;
	}

}
