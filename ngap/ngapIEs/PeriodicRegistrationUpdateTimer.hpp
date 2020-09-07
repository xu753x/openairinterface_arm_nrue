#ifndef _PERIODICREGISTRATIONUPDATETIMER_H_
#define _PERIODICREGISTRATIONUPDATETIMER_H_

extern "C"{
  #include "Ngap_PeriodicRegistrationUpdateTimer.h"
}

namespace ngap{

	class PeriodicRegistrationUpdateTimer{
	public:
		PeriodicRegistrationUpdateTimer();
		virtual ~PeriodicRegistrationUpdateTimer();
		
		void setPeriodicRegistrationUpdateTimer(uint8_t m_updatetimer);
		void getPeriodicRegistrationUpdateTimer(uint8_t &m_updatetimer);
		
		bool encode2PeriodicRegistrationUpdateTimer(Ngap_PeriodicRegistrationUpdateTimer_t *periodicRegistrationUpdateTimer);
		bool decodefromPeriodicRegistrationUpdateTimer(Ngap_PeriodicRegistrationUpdateTimer_t *periodicRegistrationUpdateTimer);
	private:
		uint8_t updatetimer;
};

}

#endif



