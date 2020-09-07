#ifndef _UERADIOCAPABILITY_H_
#define _UERADIOCAPABILITY_H_

extern "C"{
  #include "Ngap_UERadioCapability.h"
}

namespace ngap{

	class UERadioCapability{
	public:
		UERadioCapability();
		virtual ~UERadioCapability();

		bool encode2UERadioCapability(Ngap_UERadioCapability_t &ueRadioCapability);
		bool decodefromUERadioCapability(Ngap_UERadioCapability_t &ueRadioCapability);
		bool getUERadioCapability(uint8_t* &buffer,size_t &size);
		void setUERadioCapability(uint8_t* buffer,size_t size);
	private:
		char *ueRadioCapabilitybuffer;
		size_t sizeofueRadioCapabilitybuffer;
	};

}

#endif
