#ifndef _UEAGGREGATEMAXBITRATE_H_
#define _UEAGGREGATEMAXBITRATE_H_

extern "C"{
  #include "Ngap_UEAggregateMaximumBitRate.h"
}

namespace ngap{

	class UEAggregateMaxBitRate{
	public:
		UEAggregateMaxBitRate();
		virtual ~UEAggregateMaxBitRate();
		
		void setUEAggregateMaxBitRate(long bit_rate_downlink,long bit_rate_uplink);
		bool getUEAggregateMaxBitRate(long &bit_rate_downlink,long &bit_rate_uplink);		
		
		bool encode2UEAggregateMaxBitRate(Ngap_UEAggregateMaximumBitRate_t &ueAggregateMaxBitRate);
		bool decodefromUEAggregateMaxBitRate(Ngap_UEAggregateMaximumBitRate_t &ueAggregateMaxBitRate);
	private:
		long ueaggregatemaxbitratedl;
		long ueaggregatemaxbitrateul;
};

}

#endif


