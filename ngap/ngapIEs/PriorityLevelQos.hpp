#ifndef _PRIORITYLEVELQOS_H_
#define _PRIORITYLEVELQOS_H_

extern "C"{
  #include "Ngap_PriorityLevelQos.h"
}

namespace ngap{

	class PriorityLevelQos{
	public:
		PriorityLevelQos();
		virtual ~PriorityLevelQos();
		
		void setPriorityLevelQos(long value);
		bool getPriorityLevelQos(long &value);
		
		bool encode2PriorityLevelQos(Ngap_PriorityLevelQos_t *);
		bool decodefromPriorityLevelQos(Ngap_PriorityLevelQos_t *);
	private:
		long prioritylevelqos;
	};

}

#endif

