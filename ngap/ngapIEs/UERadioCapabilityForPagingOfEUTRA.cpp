#include "UERadioCapabilityForPagingOfEUTRA.hpp"

namespace ngap{

	UERadioCapabilityForPagingOfEUTRA::UERadioCapabilityForPagingOfEUTRA()
	{
		EUTRAbuffer = NULL;
		sizeofEUTRAbuffer = -1;
	}
	UERadioCapabilityForPagingOfEUTRA::~UERadioCapabilityForPagingOfEUTRA(){}

	
	bool UERadioCapabilityForPagingOfEUTRA::encode2UERadioCapabilityForPagingOfEUTRA(Ngap_UERadioCapabilityForPagingOfEUTRA_t *ueRadioCapabilityForPagingOfEUTRA)
	{
		int ret;
		ret = OCTET_STRING_fromBuf(ueRadioCapabilityForPagingOfEUTRA,EUTRAbuffer,sizeofEUTRAbuffer);
		if(ret != 0)return false;
		return true;
	}
	bool UERadioCapabilityForPagingOfEUTRA::decodefromUERadioCapabilityForPagingOfEUTRA(Ngap_UERadioCapabilityForPagingOfEUTRA_t *ueRadioCapabilityForPagingOfEUTRA)
	{
		EUTRAbuffer = (char *)ueRadioCapabilityForPagingOfEUTRA->buf;
		sizeofEUTRAbuffer = ueRadioCapabilityForPagingOfEUTRA->size;
		return true;
	}
	bool UERadioCapabilityForPagingOfEUTRA::getUERadioCapabilityForPagingOfEUTRA(uint8_t* &buffer,size_t &size)
	{
		buffer = (uint8_t *)EUTRAbuffer;
		size = sizeofEUTRAbuffer;
		if(!EUTRAbuffer) return false;
		if(sizeofEUTRAbuffer < 0) return false;

		return true;
	}
	void UERadioCapabilityForPagingOfEUTRA::setUERadioCapabilityForPagingOfEUTRA(uint8_t* buffer,size_t size)
	{
		EUTRAbuffer = (char *)buffer;
		sizeofEUTRAbuffer = size;
	}
}
