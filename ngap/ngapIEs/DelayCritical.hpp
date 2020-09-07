#ifndef _DELAYCRITICAL_H_
#define _DELAYCRITICAL_H_

extern "C"{
  #include "Ngap_DelayCritical.h"
}

namespace ngap{

	class DelayCritical{
	public:
		DelayCritical();
		virtual ~DelayCritical();
		
		void setDelayCritical(e_Ngap_DelayCritical value);
		bool getDelayCritical(e_Ngap_DelayCritical &value);
		
		bool encode2DelayCritical(Ngap_DelayCritical_t *);
		bool decodefromDelayCritical(Ngap_DelayCritical_t *);
	private:
		long delaycritical;
	};

}

#endif


