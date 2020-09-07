#ifndef _PACKETDELAYBUDGET_H_
#define _PACKETDELAYBUDGET_H_

extern "C"{
  #include "Ngap_PacketDelayBudget.h"
}

namespace ngap{

	class PacketDelayBudget{
	public:
		PacketDelayBudget();
		virtual ~PacketDelayBudget();
		
		void setPacketDelayBudget(long value);
		bool getPacketDelayBudget(long &value);
		
		bool encode2PacketDelayBudget(Ngap_PacketDelayBudget_t *);
		bool decodefromPacketDelayBudget(Ngap_PacketDelayBudget_t *);
	private:
		long packetdelaybudget;
	};

}

#endif


