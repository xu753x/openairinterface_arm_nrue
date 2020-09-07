#include "EUTRACellIdentity.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	EUTRACellIdentity::EUTRACellIdentity()
	{
		eutracellidentity = 0;
	}
	EUTRACellIdentity::~EUTRACellIdentity(){}

	void EUTRACellIdentity::setEUTRACellIdentity(uint32_t m_eutracellidentity)
	{
		eutracellidentity = m_eutracellidentity;
	}
	bool EUTRACellIdentity::encode2bitstring(Ngap_EUTRACellIdentity_t&eUTRACellIdentity)
	{
		eUTRACellIdentity.bits_unused = 4;
		eUTRACellIdentity.size = 4;
		eUTRACellIdentity.buf = (uint8_t *)calloc(1,sizeof(uint32_t));
		if(!eUTRACellIdentity.buf ) return false;
		eUTRACellIdentity.buf[3] = eutracellidentity & 0x000000ff;
		eUTRACellIdentity.buf[2] = (eutracellidentity & 0x0000ff00)>>8;
		eUTRACellIdentity.buf[1] = (eutracellidentity & 0x00ff0000)>>16;
		eUTRACellIdentity.buf[0] = (eutracellidentity & 0xff000000)>>24;

		return true;
	}
	bool EUTRACellIdentity::decodefrombitstring(Ngap_EUTRACellIdentity_t &eUTRACellIdentity)
	{
		if(!eUTRACellIdentity.buf) return false;
		
		eutracellidentity = eUTRACellIdentity.buf[0]<<24;
		eutracellidentity |= eUTRACellIdentity.buf[1]<<16;
		eutracellidentity |= eUTRACellIdentity.buf[2]<<8;
		eutracellidentity |= eUTRACellIdentity.buf[3];

		return true;
	}
	uint32_t EUTRACellIdentity::getEUTRACellIdentity()
	{
		return eutracellidentity;
	}
}
