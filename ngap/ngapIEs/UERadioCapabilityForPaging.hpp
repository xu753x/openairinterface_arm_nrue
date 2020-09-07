#ifndef _UERADIOCAPABILITYFORPAGING_H_
#define _UERADIOCAPABILITYFORPAGING_H_

#include "UERadioCapabilityForPagingOfNR.hpp"
#include "UERadioCapabilityForPagingOfEUTRA.hpp"

extern "C"{
  #include "Ngap_UERadioCapabilityForPaging.h"
}

namespace ngap{

	class UERadioCapabilityForPaging{
	public:
		UERadioCapabilityForPaging();
		virtual ~UERadioCapabilityForPaging();

		bool encode2UERadioCapabilityForPaging(Ngap_UERadioCapabilityForPaging_t *ueRadioCapabilityForPaging);
		bool decodefromUERadioCapabilityForPaging(Ngap_UERadioCapabilityForPaging_t *ueRadioCapabilityForPaging);
		bool getUERadioCapabilityForPaging(UERadioCapabilityForPagingOfNR *&m_ueRadioCapabilityForPagingOfNR,UERadioCapabilityForPagingOfEUTRA *&m_ueRadioCapabilityForPagingOfEUTRA);
		void setUERadioCapabilityForPaging(UERadioCapabilityForPagingOfNR *m_ueRadioCapabilityForPagingOfNR,UERadioCapabilityForPagingOfEUTRA *m_ueRadioCapabilityForPagingOfEUTRA);
	private:
		UERadioCapabilityForPagingOfNR *ueRadioCapabilityForPagingOfNR;
		UERadioCapabilityForPagingOfEUTRA *ueRadioCapabilityForPagingOfEUTRA;
	};

}

#endif
