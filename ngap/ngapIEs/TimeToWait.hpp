#ifndef _TIMETOWAIT_H_
#define _TIMETOWAIT_H_

extern "C"{
  #include "Ngap_NGSetupFailure.h"
  #include "Ngap_ProtocolIE-Field.h"
  #include "Ngap_TimeToWait.h"
}

namespace ngap{

	class TimeToWait{
	public:
		TimeToWait();
		virtual ~TimeToWait();
		
		bool encode2TimeToWait(Ngap_TimeToWait_t *);
		bool decodefromTimeToWait(Ngap_TimeToWait_t *);
		void setValue(e_Ngap_TimeToWait);
		long getValue();
	private:
		long timeValue;
	};
}

#endif

