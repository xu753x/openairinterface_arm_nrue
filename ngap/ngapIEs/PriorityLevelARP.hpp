#ifndef _PRIORITYLEVELARP_H_
#define _PRIORITYLEVELARP_H_

extern "C"{
  #include "Ngap_PriorityLevelARP.h"
}

namespace ngap{

	class PriorityLevelARP{
	public:
		PriorityLevelARP();
		virtual ~PriorityLevelARP();
		
		void setPriorityLevelARP(long value);
		bool getPriorityLevelARP(long &value);
		
		bool encode2PriorityLevelARP(Ngap_PriorityLevelARP_t *);
		bool decodefromPriorityLevelARP(Ngap_PriorityLevelARP_t *);
	private:
		long prioritylevelarp;
	};

}

#endif


