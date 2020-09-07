#ifndef _PDUSESSIONAGGREGATEMAXIMUMBITRATE_H_
#define _PDUSESSIONAGGREGATEMAXIMUMBITRATE_H_

extern "C"{
  #include "Ngap_PDUSessionAggregateMaximumBitRate.h"
}

namespace ngap{

	class PduSessionAggregateMaximumBitRate{
	public:
		PduSessionAggregateMaximumBitRate();
		virtual ~PduSessionAggregateMaximumBitRate();
		
		void setPduSessionAggregateMaximumBitRate(long bit_rate_downlink,long bit_rate_uplink);
		bool getPduSessionAggregateMaximumBitRate(long &bit_rate_downlink,long &bit_rate_uplink);		
		
		bool encode2PduSessionAggregateMaximumBitRate(Ngap_PDUSessionAggregateMaximumBitRate_t &pdusessionAggregateMaxBitRate);
		bool decodefromPduSessionAggregateMaximumBitRate(Ngap_PDUSessionAggregateMaximumBitRate_t &pdusessionAggregateMaxBitRate);
	private:
		long aggregatemaxbitratedl;
		long aggregatemaxbitrateul;
};

}

#endif
