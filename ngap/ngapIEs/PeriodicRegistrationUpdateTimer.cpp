#include "PeriodicRegistrationUpdateTimer.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PeriodicRegistrationUpdateTimer::PeriodicRegistrationUpdateTimer(){}
	PeriodicRegistrationUpdateTimer::~PeriodicRegistrationUpdateTimer(){}
	
	void PeriodicRegistrationUpdateTimer::setPeriodicRegistrationUpdateTimer(uint8_t m_updatetimer)
	{
		updatetimer = m_updatetimer;
	}
	bool PeriodicRegistrationUpdateTimer::encode2PeriodicRegistrationUpdateTimer(Ngap_PeriodicRegistrationUpdateTimer_t *periodicRegistrationUpdateTimer)
	{
		periodicRegistrationUpdateTimer->size = sizeof(uint8_t);
		periodicRegistrationUpdateTimer->bits_unused = 0;
		periodicRegistrationUpdateTimer->buf = (uint8_t *)calloc(1, periodicRegistrationUpdateTimer->size);
		if(!periodicRegistrationUpdateTimer->buf) return false;
		periodicRegistrationUpdateTimer->buf[0] = updatetimer;
		
		return true; 
	}
	bool PeriodicRegistrationUpdateTimer::decodefromPeriodicRegistrationUpdateTimer(Ngap_PeriodicRegistrationUpdateTimer_t *periodicRegistrationUpdateTimer)
	{
		if(!periodicRegistrationUpdateTimer->buf) return false;
		updatetimer = periodicRegistrationUpdateTimer->buf[0];
		
		return true;
	}
	void PeriodicRegistrationUpdateTimer::getPeriodicRegistrationUpdateTimer(uint8_t &m_updatetimer)
	{
		m_updatetimer = updatetimer;
	}

}


