#ifndef _GNB_ID_H_
#define _GNB_ID_H_

extern "C"{
  #include "Ngap_GNB-ID.h"
}

namespace ngap{

	class GNB_ID{
	public:
		GNB_ID();
		virtual ~GNB_ID(); 

		bool encode2bitstring(Ngap_GNB_ID_t &);
		bool decodefrombitstring(Ngap_GNB_ID_t &);
		void setValue(uint32_t gnbId);
		long getValue();
	private:
		struct gNBId_s{
		uint32_t id;
		uint8_t  biteslen;
		} gNBId;//22bits to 32bits
	};

}

#endif
