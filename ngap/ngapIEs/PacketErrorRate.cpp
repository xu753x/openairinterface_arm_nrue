#include "PacketErrorRate.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	PacketErrorRate::PacketErrorRate(){}  

	PacketErrorRate::~PacketErrorRate(){} 

	void PacketErrorRate::setPacketErrorRate(long m_pERScalar,long m_pERExponent)
	{
		pERScalar = m_pERScalar;
		pERExponent = m_pERExponent;
	}
	bool PacketErrorRate::getPacketErrorRate(long &m_pERScalar,long &m_pERExponent)
	{
		m_pERScalar = pERScalar;
		m_pERExponent = pERExponent;
		
		return true;
	}
	
	bool PacketErrorRate::encode2PacketErrorRate(Ngap_PacketErrorRate_t *packetErrorRate)
	{
		packetErrorRate->pERScalar = pERScalar;
		packetErrorRate->pERExponent = pERExponent;
	
		return true;
	}
	bool PacketErrorRate::decodefromPacketErrorRate(Ngap_PacketErrorRate_t *packetErrorRate)
	{
		pERScalar = packetErrorRate->pERScalar;
		pERExponent = packetErrorRate->pERExponent;
		
		return true;
	}
}

