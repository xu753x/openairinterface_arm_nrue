#ifndef _RELATIVEAMFCAPACITY_H_
#define _RELATIVEAMFCAPACITY_H_
	
extern "C"{
	#include "Ngap_RelativeAMFCapacity.h"
}

namespace ngap{

	class RelativeAMFCapacity{
	public:
		RelativeAMFCapacity();
		virtual ~RelativeAMFCapacity();

		bool encode2RelativeAMFCapacity(Ngap_RelativeAMFCapacity_t *);
		bool decodefromRelativeAMFCapacity(Ngap_RelativeAMFCapacity_t *);
		void setValue(long m_amfcapacity);
		long getValue();
	private:
		long amfcapacity;
	};
}

#endif
