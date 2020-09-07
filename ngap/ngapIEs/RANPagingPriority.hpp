#ifndef _RANPAGINGPRIORITY_H_
#define _RANPAGINGPRIORITY_H_

extern "C"{
  #include "Ngap_RANPagingPriority.h"
}

namespace ngap{

	class RANPagingPriority{
	public:
		RANPagingPriority();
		virtual ~RANPagingPriority();
		
		void setRANPagingPriority(uint8_t);
		uint8_t getRANPagingPriority();
		
		bool encode2RANPagingPriority(Ngap_RANPagingPriority_t &);
		bool decodefromRANPagingPriority(Ngap_RANPagingPriority_t &);
	private:
		uint8_t ranPagingPriority;
	};

}

#endif


