#ifndef _UERADIOCAPABILITYFORPAGINGOFNR_H_
#define _UERADIOCAPABILITYFORPAGINGOFNR_H_

extern "C"{
  #include "Ngap_UERadioCapabilityForPagingOfNR.h"
}

namespace ngap{

	class UERadioCapabilityForPagingOfNR{
	public:
		UERadioCapabilityForPagingOfNR();
		virtual ~UERadioCapabilityForPagingOfNR();

		bool encode2UERadioCapabilityForPagingOfNR(Ngap_UERadioCapabilityForPagingOfNR_t *ueRadioCapabilityForPagingOfNR);
		bool decodefromUERadioCapabilityForPagingOfNR(Ngap_UERadioCapabilityForPagingOfNR_t *ueRadioCapabilityForPagingOfNR);
		bool getUERadioCapabilityForPagingOfNR(uint8_t* &buffer,size_t &size);
		void setUERadioCapabilityForPagingOfNR(uint8_t* buffer,size_t size);
	private:
		char *nRbuffer;
		size_t sizeofnRbuffer;
	};

}

#endif
