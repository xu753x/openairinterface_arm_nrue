#include "UserLocationInformationEUTRA.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	UserLocationInformationEUTRA::UserLocationInformationEUTRA()
	{
		eUTRA_CGI = NULL;
		tAI = NULL;
		/*istimeStampSet = false;
		timeStamp = NULL;*/
	}
	UserLocationInformationEUTRA::~UserLocationInformationEUTRA(){}
	
	void UserLocationInformationEUTRA::setInformationEUTRA(EUTRA_CGI* m_eUTRA_CGI,TAI* m_tAI)
	{
		eUTRA_CGI = m_eUTRA_CGI;
		tAI = m_tAI;
	}
	/*void UserLocationInformationEUTRA::setInformationEUTRA(EUTRA_CGI* m_eUTRA_CGI,TAI* m_tAI,TimeStamp* m_timeStamp)
	{
		eUTRA_CGI = m_eUTRA_CGI;
		tAI = m_tAI;
		istimeStampSet = true;
		timeStamp = m_timeStamp;
	}*/
	bool UserLocationInformationEUTRA::encode2UserLocationInformationEUTRA(Ngap_UserLocationInformationEUTRA_t *userLocationInformationEUTRA)
	{
		if(!eUTRA_CGI->encode2EUTRA_CGI(&userLocationInformationEUTRA->eUTRA_CGI))
		{
			cout<<"[Warning] eUTRA_CGI->encode2EUTRA_CGI() error!"<<endl;
			return false;
		}
		if(!tAI->encode2TAI(&userLocationInformationEUTRA->tAI))
		{
			cout<<"[Warning] tAI->encode2TAI() error!"<<endl;
			return false;
		}
		#if 0
		if(istimeStampSet)
		{
			Ngap_TimeStamp_t *ieTimeStamp = (Ngap_TimeStamp_t *)calloc(1,sizeof(Ngap_TimeStamp_t));
			if(!timeStamp->encodefromTimeStamp(ieTimeStamp))
			{
				cout<<"[Warning] timeStamp->encodefromTimeStamp() error!"<<endl;
				free(ieTimeStamp);
				return false;
			}
			userLocationInformationEUTRA->timeStamp = ieTimeStamp;
		}
		#endif
		return true;
	}
	bool UserLocationInformationEUTRA::decodefromUserLocationInformationEUTRA(Ngap_UserLocationInformationEUTRA_t *userLocationInformationEUTRA)
	{
		eUTRA_CGI = new EUTRA_CGI();
		if(!eUTRA_CGI->decodefromEUTRA_CGI(&userLocationInformationEUTRA->eUTRA_CGI))
		{
			cout<<"[Warning] eUTRA_CGI->decodefromEUTRA_CGI() error!"<<endl;
			return false;
		}
		tAI = new TAI();
		if(!tAI->decodefromTAI(&userLocationInformationEUTRA->tAI))
		{
			cout<<"[Warning] tAI->decodefromTAI() error!"<<endl;
			return false;
		}
		#if 0
		if(userLocationInformationEUTRA->timeStamp)
		{
			istimeStampSet = true;
			timeStamp = new TimeStamp();
			if(!timeStamp->decodefromTimeStamp(userLocationInformationEUTRA->timeStamp))
			{
				cout<<"[Warning] timeStamp->decodefromTimeStamp() error!"<<endl;
				return false;
			}
		}
		#endif
		return true;
	}
	/*bool UserLocationInformationEUTRA::getTimeStampPresence()
	{
		return istimeStampSet;
	}*/
	/*void UserLocationInformationEUTRA::getInformationEUTRA(EUTRA_CGI* &m_eUTRA_CGI,TAI* &m_tAI,TimeStamp* &m_timeStamp)
	{
		m_eUTRA_CGI = eUTRA_CGI;
		m_tAI = tAI;
		m_timeStamp = timeStamp;
	}*/
	void UserLocationInformationEUTRA::getInformationEUTRA(EUTRA_CGI* &m_eUTRA_CGI,TAI* &m_tAI)
	{
		m_eUTRA_CGI = eUTRA_CGI;
		m_tAI = tAI;
	}
}
