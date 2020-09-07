#ifndef _FIVEQI_H_
#define _FIVEQI_H_

extern "C"{
  #include "Ngap_FiveQI.h"
}

namespace ngap{

	class FiveQI{
	public:
		FiveQI();
		virtual ~FiveQI();
		
		void setFiveQI(long value);
		bool getFiveQI(long &value);
		
		bool encode2FiveQI(Ngap_FiveQI_t *);
		bool decodefromFiveQI(Ngap_FiveQI_t *);
	private:
		long fiveqi;
	};

}

#endif
