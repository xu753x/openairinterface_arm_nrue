#ifndef _CORENETWORKASSISTANCEINFORMATION_H_
#define _CORENETWORKASSISTANCEINFORMATION_H_

#include "UEIdentityIndexValue.hpp"
#include "DefaultPagingDRX.hpp"
#include "PeriodicRegistrationUpdateTimer.hpp"
#include "MICOModeIndication.hpp"
#include "TAI.hpp"

extern "C"{
  #include "Ngap_CoreNetworkAssistanceInformation.h"
}

namespace ngap{

	class CoreNetworkAssistanceInfo{
	public:
		CoreNetworkAssistanceInfo();
		virtual ~CoreNetworkAssistanceInfo();
		
		void setCoreNetworkAssistanceInfo(UEIdentityIndexValue *m_ueIdentityIndexValue, DefaultPagingDRX *m_pagingDRX, PeriodicRegistrationUpdateTimer *m_periodicRegUpdateTimer, bool m_micoModeInd, TAI *m_tai,int m_numoftai);
		void setCoreNetworkAssistanceInfo(UEIdentityIndexValue *m_ueIdentityIndexValue, PeriodicRegistrationUpdateTimer *m_periodicRegUpdateTimer, bool m_micoModeInd, TAI *m_tai,int m_numoftai);
		void getCoreNetworkAssistanceInfo(UEIdentityIndexValue *&m_ueIdentityIndexValue, DefaultPagingDRX *&m_pagingDRX, PeriodicRegistrationUpdateTimer *&m_periodicRegUpdateTimer, bool &m_micoModeInd, TAI *&m_tai,int &m_numoftai);
		
		bool encode2CoreNetworkAssistanceInfo(Ngap_CoreNetworkAssistanceInformation_t *coreNetworkAssistanceInformation);
		bool decodefromCoreNetworkAssistanceInfo(Ngap_CoreNetworkAssistanceInformation_t *coreNetworkAssistanceInformation);
	private:
		UEIdentityIndexValue *ueIdentityIndexValue;
		DefaultPagingDRX *pagingDRX;
		PeriodicRegistrationUpdateTimer *periodicRegUpdateTimer;
		MICOModeIndication *micoModeInd;
		TAI *tai;
		int numoftai;
};

}

#endif

