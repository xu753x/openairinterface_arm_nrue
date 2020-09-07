#ifndef _USERLOCATIONINFORMATION_H_
#define _USERLOCATIONINFORMATION_H_

#include "UserLocationInformationEUTRA.hpp"
#include "UserLocationInformationNR.hpp"
#include "UserLocationInformationN3IWF.hpp"

extern "C"{
  #include "Ngap_UserLocationInformation.h"
}

namespace ngap{
	class UserLocationInformation{
	public:
		UserLocationInformation();
		virtual ~UserLocationInformation();
		
		void setInformation(UserLocationInformationEUTRA*);
		void setInformation(UserLocationInformationNR*);
		//void setInformation(UserLocationInformationN3IWF*);
		bool encodefromUserLocationInformation(Ngap_UserLocationInformation_t *userLocationInformation);
  		bool decodefromUserLocationInformation(Ngap_UserLocationInformation_t *userLocationInformation);
		Ngap_UserLocationInformation_PR getChoiceOfUserLocationInformation();
  		void getInformation(UserLocationInformationEUTRA*&);
		void getInformation(UserLocationInformationNR*&);
		//void getInformation(UserLocationInformationN3IWF*&);
		
	private:
		Ngap_UserLocationInformation_PR informationPresent;
		UserLocationInformationEUTRA *userLocationInformationEUTRA;
		UserLocationInformationNR *userLocationInformationNR;
		//UserLocationInformationN3IWF *userLocationInformationN3IWF;
	};

}

#endif