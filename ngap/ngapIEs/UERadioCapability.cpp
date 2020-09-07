#include "UERadioCapability.hpp"

namespace ngap{

	UERadioCapability::UERadioCapability()
	{
		ueRadioCapabilitybuffer = NULL;
		sizeofueRadioCapabilitybuffer = -1;
	}
	UERadioCapability::~UERadioCapability(){}
	
	bool UERadioCapability::encode2UERadioCapability(Ngap_UERadioCapability_t &ueRadioCapability)
	{
		int ret;
		ret = OCTET_STRING_fromBuf(&ueRadioCapability,ueRadioCapabilitybuffer,sizeofueRadioCapabilitybuffer);
		if(ret != 0)return false;
		return true;
	}
	bool UERadioCapability::decodefromUERadioCapability(Ngap_UERadioCapability_t &ueRadioCapability)
	{
		ueRadioCapabilitybuffer = (char *)ueRadioCapability.buf;
		sizeofueRadioCapabilitybuffer = ueRadioCapability.size;
		return true;
	}
	bool UERadioCapability::getUERadioCapability(uint8_t* &buffer,size_t &size)
	{
		buffer = (uint8_t *)ueRadioCapabilitybuffer;
		size = sizeofueRadioCapabilitybuffer;
		if(!ueRadioCapabilitybuffer) return false;
		if(sizeofueRadioCapabilitybuffer < 0) return false;

		return true;
	}
	void UERadioCapability::setUERadioCapability(uint8_t* buffer,size_t size)
	{
		ueRadioCapabilitybuffer = (char *)buffer;
		sizeofueRadioCapabilitybuffer = size;
	}
}
