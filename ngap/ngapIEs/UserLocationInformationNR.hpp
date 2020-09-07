#ifndef _USERLOCATIONINFORMATIONNR_H_
#define _USERLOCATIONINFORMATIONNR_H_

#include "NR-CGI.hpp"
#include "TAI.hpp"
//#include "TimeStamp.hpp"

extern "C"{
  #include "Ngap_UserLocationInformationNR.h"
}

namespace ngap{
	class UserLocationInformationNR{
	public:
		UserLocationInformationNR();
		virtual ~UserLocationInformationNR();
		
		void setInformationNR(NR_CGI* ,TAI* );
		//void setInformationEUTRA(EUTRA_CGI* m_eUTRA_CGI,TAI* m_tAI,TimeStamp* m_timeStamp);
		bool encode2UserLocationInformationNR(Ngap_UserLocationInformationNR_t *);
  		bool decodefromUserLocationInformationNR(Ngap_UserLocationInformationNR_t *);
		//void getInformationEUTRA(EUTRA_CGI* &m_eUTRA_CGI,TAI* &m_tAI,TimeStamp* &m_timeStamp);
		void getInformationNR(NR_CGI* &,TAI* &);
		//bool getTimeStampPresence();
		
	private:
		NR_CGI *nR_CGI;
		TAI *tAI;
		//bool istimeStampSet;
		//TimeStamp *timeStamp;
	};

}

#endif
