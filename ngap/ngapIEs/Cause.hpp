#ifndef _CAUSE_H_
#define _CAUSE_H_

extern "C"{
  #include "Ngap_Cause.h"
}

namespace ngap{

	class Cause{
	public:
		Cause();
		virtual ~Cause();
		
		bool encode2Cause(Ngap_Cause_t *cause);
		bool decodefromCause(Ngap_Cause_t * pdu);
		void setChoiceOfCause(Ngap_Cause_PR m_causePresent);
		Ngap_Cause_PR getChoiceOfCause();
		void setValue(long);
		long getValue();
	private:
		long causeValue;
		Ngap_Cause_PR causePresent;
	};
}

#endif