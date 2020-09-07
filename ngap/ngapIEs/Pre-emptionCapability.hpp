#ifndef _PRE_EMPTIONCAPABILITY_H_
#define _PRE_EMPTIONCAPABILITY_H_

extern "C"{
  #include "Ngap_Pre-emptionCapability.h"
}

namespace ngap{

	class Pre_emptionCapability{
	public:
		Pre_emptionCapability();
		virtual ~Pre_emptionCapability();
		
		void setPre_emptionCapability(e_Ngap_Pre_emptionCapability value);
		bool getPre_emptionCapability(e_Ngap_Pre_emptionCapability &value);
		
		bool encode2Pre_emptionCapability(Ngap_Pre_emptionCapability_t *);
		bool decodefromPre_emptionCapability(Ngap_Pre_emptionCapability_t *);
	private:
		long capability;
	};

}

#endif


