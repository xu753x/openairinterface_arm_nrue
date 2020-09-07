#include "UERadioCapabilityForPagingOfNR.hpp"

namespace ngap{

	UERadioCapabilityForPagingOfNR::UERadioCapabilityForPagingOfNR()
	{
		nRbuffer = NULL;
		sizeofnRbuffer = -1;
	}
	UERadioCapabilityForPagingOfNR::~UERadioCapabilityForPagingOfNR(){}

	
	bool UERadioCapabilityForPagingOfNR::encode2UERadioCapabilityForPagingOfNR(Ngap_UERadioCapabilityForPagingOfNR_t *ueRadioCapabilityForPagingOfNR)
	{
		int ret;
		ret = OCTET_STRING_fromBuf(ueRadioCapabilityForPagingOfNR,nRbuffer,sizeofnRbuffer);
		if(ret != 0)return false;
		return true;
	}
	bool UERadioCapabilityForPagingOfNR::decodefromUERadioCapabilityForPagingOfNR(Ngap_UERadioCapabilityForPagingOfNR_t *ueRadioCapabilityForPagingOfNR)
	{
		nRbuffer = (char *)ueRadioCapabilityForPagingOfNR->buf;
		sizeofnRbuffer = ueRadioCapabilityForPagingOfNR->size;
		return true;
	}
	bool UERadioCapabilityForPagingOfNR::getUERadioCapabilityForPagingOfNR(uint8_t* &buffer,size_t &size)
	{
		buffer = (uint8_t *)nRbuffer;
		size = sizeofnRbuffer;
		if(!nRbuffer) return false;
		if(sizeofnRbuffer < 0) return false;

		return true;
	}
	void UERadioCapabilityForPagingOfNR::setUERadioCapabilityForPagingOfNR(uint8_t* buffer,size_t size)
	{
		nRbuffer = (char *)buffer;
		sizeofnRbuffer = size;
	}
}
