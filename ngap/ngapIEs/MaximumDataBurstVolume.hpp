#ifndef _MAXIMUMDATABURSTVOLUME_H_
#define _MAXIMUMDATABURSTVOLUME_H_

extern "C"{
  #include "Ngap_MaximumDataBurstVolume.h"
}

namespace ngap{

	class MaximumDataBurstVolume{
	public:
		MaximumDataBurstVolume();
		virtual ~MaximumDataBurstVolume();
		
		void setMaximumDataBurstVolume(long value);
		bool getMaximumDataBurstVolume(long &value);
		
		bool encode2MaximumDataBurstVolume(Ngap_MaximumDataBurstVolume_t *);
		bool decodefromMaximumDataBurstVolume(Ngap_MaximumDataBurstVolume_t *);
	private:
		long maximumdataburstvolume;
	};

}

#endif



