#include "UERadioCapabilityForPaging.hpp"

namespace ngap{

	UERadioCapabilityForPaging::UERadioCapabilityForPaging()
	{
		ueRadioCapabilityForPagingOfNR = NULL;
		ueRadioCapabilityForPagingOfEUTRA = NULL;
	}
	UERadioCapabilityForPaging::~UERadioCapabilityForPaging(){}

	
	bool UERadioCapabilityForPaging::encode2UERadioCapabilityForPaging(Ngap_UERadioCapabilityForPaging_t *ueRadioCapabilityForPaging)
	{
		if(ueRadioCapabilityForPagingOfNR)
		{
			ueRadioCapabilityForPaging->uERadioCapabilityForPagingOfNR = (Ngap_UERadioCapabilityForPagingOfNR_t *)calloc(1,sizeof(Ngap_UERadioCapabilityForPagingOfNR_t));
			if(!ueRadioCapabilityForPagingOfNR->encode2UERadioCapabilityForPagingOfNR(ueRadioCapabilityForPaging->uERadioCapabilityForPagingOfNR)) return false;
		}
		if(ueRadioCapabilityForPagingOfEUTRA)
		{
			ueRadioCapabilityForPaging->uERadioCapabilityForPagingOfEUTRA = (Ngap_UERadioCapabilityForPagingOfEUTRA_t *)calloc(1,sizeof(Ngap_UERadioCapabilityForPagingOfEUTRA_t));
			if(!ueRadioCapabilityForPagingOfEUTRA->encode2UERadioCapabilityForPagingOfEUTRA(ueRadioCapabilityForPaging->uERadioCapabilityForPagingOfEUTRA)) return false;
		}
	
		return true;
	}
	bool UERadioCapabilityForPaging::decodefromUERadioCapabilityForPaging(Ngap_UERadioCapabilityForPaging_t *ueRadioCapabilityForPaging)
	{
		if(ueRadioCapabilityForPaging->uERadioCapabilityForPagingOfNR)
		{
			ueRadioCapabilityForPagingOfNR = new UERadioCapabilityForPagingOfNR();
			if(!ueRadioCapabilityForPagingOfNR->decodefromUERadioCapabilityForPagingOfNR(ueRadioCapabilityForPaging->uERadioCapabilityForPagingOfNR)) return false;
		}
		if(ueRadioCapabilityForPaging->uERadioCapabilityForPagingOfEUTRA)
		{
			ueRadioCapabilityForPagingOfEUTRA = new UERadioCapabilityForPagingOfEUTRA();
			if(!ueRadioCapabilityForPagingOfEUTRA->decodefromUERadioCapabilityForPagingOfEUTRA(ueRadioCapabilityForPaging->uERadioCapabilityForPagingOfEUTRA)) return false;
		}
	
		return true;
	}
	bool UERadioCapabilityForPaging::getUERadioCapabilityForPaging(UERadioCapabilityForPagingOfNR *&m_ueRadioCapabilityForPagingOfNR,UERadioCapabilityForPagingOfEUTRA *&m_ueRadioCapabilityForPagingOfEUTRA)
	{
		m_ueRadioCapabilityForPagingOfNR = ueRadioCapabilityForPagingOfNR;
		m_ueRadioCapabilityForPagingOfEUTRA = ueRadioCapabilityForPagingOfEUTRA;

		return true;
	}
	void UERadioCapabilityForPaging::setUERadioCapabilityForPaging(UERadioCapabilityForPagingOfNR *m_ueRadioCapabilityForPagingOfNR,UERadioCapabilityForPagingOfEUTRA *m_ueRadioCapabilityForPagingOfEUTRA)
	{
		ueRadioCapabilityForPagingOfNR = m_ueRadioCapabilityForPagingOfNR;
		ueRadioCapabilityForPagingOfEUTRA = m_ueRadioCapabilityForPagingOfEUTRA;
	}
}
