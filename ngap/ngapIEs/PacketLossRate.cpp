#include "PacketLossRate.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	PacketLossRate::PacketLossRate(){}  

	PacketLossRate::~PacketLossRate(){} 

	void PacketLossRate::setPacketLossRate(long value)
	{
		packetlossrate = value;
	}
	bool PacketLossRate::getPacketLossRate(long &value)
	{
		value = packetlossrate;
		
		return true;
	}
	
	bool PacketLossRate::encode2PacketLossRate(Ngap_PacketLossRate_t *packetLossRate)
	{
		*packetLossRate = packetlossrate;
	
		return true;
	}
	bool PacketLossRate::decodefromPacketLossRate(Ngap_PacketLossRate_t *packetLossRate)
	{
		packetlossrate = *packetLossRate;
		
		return true;
	}
}
