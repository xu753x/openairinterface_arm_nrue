#include "TimeToWait.hpp"

#include<iostream>
using namespace std;

namespace ngap{

	TimeToWait::TimeToWait()
	{
		timeValue = -1;
	}
	TimeToWait::~TimeToWait(){}
	
	void TimeToWait::setValue(e_Ngap_TimeToWait m_timeToWait)
	{
		timeValue = m_timeToWait;
	}

	bool TimeToWait::encode2TimeToWait(Ngap_TimeToWait_t * timeToWait)
	{
		*timeToWait = timeValue;
		return true;
	}
	
	bool TimeToWait::decodefromTimeToWait(Ngap_TimeToWait_t * pdu)
	{
		timeValue = *pdu;
		return true;
	}
	long TimeToWait::getValue()
	{
		return timeValue;
	}
}

