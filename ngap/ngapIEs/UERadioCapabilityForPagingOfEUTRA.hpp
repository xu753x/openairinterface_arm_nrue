#ifndef _UERADIOCAPABILITYFORPAGINGOFEUTRA_H_
#define _UERADIOCAPABILITYFORPAGINGOFEUTRA_H_

extern "C"{
  #include "Ngap_UERadioCapabilityForPagingOfEUTRA.h"
}

namespace ngap{

	class UERadioCapabilityForPagingOfEUTRA{
	public:
		UERadioCapabilityForPagingOfEUTRA();
		virtual ~UERadioCapabilityForPagingOfEUTRA();

		bool encode2UERadioCapabilityForPagingOfEUTRA(Ngap_UERadioCapabilityForPagingOfEUTRA_t *ueRadioCapabilityForPagingOfEUTRA);
		bool decodefromUERadioCapabilityForPagingOfEUTRA(Ngap_UERadioCapabilityForPagingOfEUTRA_t *ueRadioCapabilityForPagingOfEUTRA);
		bool getUERadioCapabilityForPagingOfEUTRA(uint8_t* &buffer,size_t &size);
		void setUERadioCapabilityForPagingOfEUTRA(uint8_t* buffer,size_t size);
	private:
		char *EUTRAbuffer;
		size_t sizeofEUTRAbuffer;
	};

}

#endif
