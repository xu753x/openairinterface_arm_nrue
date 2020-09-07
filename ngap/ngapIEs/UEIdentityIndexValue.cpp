#include "UEIdentityIndexValue.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	UEIdentityIndexValue::UEIdentityIndexValue(){}
	UEIdentityIndexValue::~UEIdentityIndexValue(){}
	
	void UEIdentityIndexValue::setUEIdentityIndexValue(uint16_t m_indexLength10/*10bits限制*/)
	{
		indexLength10 = m_indexLength10;
	}
	bool UEIdentityIndexValue::encode2UEIdentityIndexValue(Ngap_UEIdentityIndexValue_t *ueIdentityIndexValue)
	{
		ueIdentityIndexValue->present = Ngap_UEIdentityIndexValue_PR_indexLength10;
		ueIdentityIndexValue->choice.indexLength10.size = sizeof(uint16_t);
		ueIdentityIndexValue->choice.indexLength10.bits_unused = 6;
		ueIdentityIndexValue->choice.indexLength10.buf = (uint8_t *)calloc(1, ueIdentityIndexValue->choice.indexLength10.size);
		if(!ueIdentityIndexValue->choice.indexLength10.buf) return false;
		ueIdentityIndexValue->choice.indexLength10.buf[0] = (indexLength10>>8) & 0x03;
		ueIdentityIndexValue->choice.indexLength10.buf[1] = indexLength10 & 0xff;
		
		return true; 
	}
	bool UEIdentityIndexValue::decodefromUEIdentityIndexValue(Ngap_UEIdentityIndexValue_t *ueIdentityIndexValue)
	{
		if(ueIdentityIndexValue->present != Ngap_UEIdentityIndexValue_PR_indexLength10) return false;
		if(!ueIdentityIndexValue->choice.indexLength10.buf) return false;
		indexLength10 = ueIdentityIndexValue->choice.indexLength10.buf[0];
		indexLength10 = indexLength10 << 8;
		indexLength10 |= ueIdentityIndexValue->choice.indexLength10.buf[1];
		
		return true;
	}
	void UEIdentityIndexValue::getUEIdentityIndexValue(uint16_t &m_indexLength10/*10bits限制*/)
	{
		m_indexLength10 = indexLength10;
	}

}

