#ifndef _PACKETLOSSRATE_H_
#define _PACKETLOSSRATE_H_

extern "C"{
  #include "Ngap_PacketLossRate.h"
}

namespace ngap{

	class PacketLossRate{
	public:
		PacketLossRate();
		virtual ~PacketLossRate();
		
		void setPacketLossRate(long value);
		bool getPacketLossRate(long &value);
		
		bool encode2PacketLossRate(Ngap_PacketLossRate_t *);
		bool decodefromPacketLossRate(Ngap_PacketLossRate_t *);
	private:
		long packetlossrate;
	};

}

#endif


