#ifndef _USERLOCATIONINFORMATIONEUTRA_H_
#define _USERLOCATIONINFORMATIONEUTRA_H_

#include "EUTRA-CGI.hpp"
#include "TAI.hpp"
//#include "TimeStamp.hpp"

extern "C"{
  #include "Ngap_UserLocationInformationEUTRA.h"
}

namespace ngap{
	class UserLocationInformationEUTRA{
	public:
		UserLocationInformationEUTRA();
		virtual ~UserLocationInformationEUTRA();
		
		void setInformationEUTRA(EUTRA_CGI* m_eUTRA_CGI,TAI* m_tAI);
		//void setInformationEUTRA(EUTRA_CGI* m_eUTRA_CGI,TAI* m_tAI,TimeStamp* m_timeStamp);
		bool encode2UserLocationInformationEUTRA(Ngap_UserLocationInformationEUTRA_t *userLocationInformation);
  		bool decodefromUserLocationInformationEUTRA(Ngap_UserLocationInformationEUTRA_t *userLocationInformation);
		//void getInformationEUTRA(EUTRA_CGI* &m_eUTRA_CGI,TAI* &m_tAI,TimeStamp* &m_timeStamp);
		void getInformationEUTRA(EUTRA_CGI* &m_eUTRA_CGI,TAI* &m_tAI);
		//bool getTimeStampPresence();
		
	private:
		EUTRA_CGI *eUTRA_CGI;
		TAI *tAI;
		//bool istimeStampSet;
		//TimeStamp *timeStamp;
	};

}

#endif

