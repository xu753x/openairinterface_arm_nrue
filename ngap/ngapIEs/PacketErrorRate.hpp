#ifndef _PACKETERRORRATE_H_
#define _PACKETERRORRATE_H_

extern "C"{
  #include "Ngap_PacketErrorRate.h"
}

namespace ngap{

	class PacketErrorRate{
	public:
		PacketErrorRate();
		virtual ~PacketErrorRate();
		
		void setPacketErrorRate(long m_pERScalar,long m_pERExponent);
		bool getPacketErrorRate(long &m_pERScalar,long &m_pERExponent);
		
		bool encode2PacketErrorRate(Ngap_PacketErrorRate_t *);
		bool decodefromPacketErrorRate(Ngap_PacketErrorRate_t *);
	private:
		long pERScalar;
		long pERExponent;
	};

}

#endif



