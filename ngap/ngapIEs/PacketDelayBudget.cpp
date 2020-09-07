#include "PacketDelayBudget.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	PacketDelayBudget::PacketDelayBudget(){}  

	PacketDelayBudget::~PacketDelayBudget(){} 

	void PacketDelayBudget::setPacketDelayBudget(long value)
	{
		packetdelaybudget = value;
	}
	bool PacketDelayBudget::getPacketDelayBudget(long &value)
	{
		value = packetdelaybudget;
		
		return true;
	}
	
	bool PacketDelayBudget::encode2PacketDelayBudget(Ngap_PacketDelayBudget_t *packetDelayBudget)
	{
		*packetDelayBudget = packetdelaybudget;
	
		return true;
	}
	bool PacketDelayBudget::decodefromPacketDelayBudget(Ngap_PacketDelayBudget_t *packetDelayBudget)
	{
		packetdelaybudget = *packetDelayBudget;
		
		return true;
	}
}
