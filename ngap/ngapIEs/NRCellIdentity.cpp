#include "NRCellIdentity.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	NRCellIdentity::NRCellIdentity()
	{
		nrcellidentity = 0;
	}
	NRCellIdentity::~NRCellIdentity(){}

	void NRCellIdentity::setNRCellIdentity(unsigned long m_nrcellidentity)
	{
		nrcellidentity = m_nrcellidentity;
	}
	bool NRCellIdentity::encode2bitstring(Ngap_NRCellIdentity_t&nRCellIdentity)
	{
		nRCellIdentity.bits_unused = 4;
		nRCellIdentity.size = 5;
		nRCellIdentity.buf = (uint8_t *)calloc(1,sizeof(uint32_t)+sizeof(uint8_t));
		if(!nRCellIdentity.buf ) return false;
		nRCellIdentity.buf[4] = nrcellidentity & 0x00000000ff;
		nRCellIdentity.buf[3] = (nrcellidentity & 0x000000ff00)>>8;
		nRCellIdentity.buf[2] = (nrcellidentity & 0x0000ff0000)>>16;
		nRCellIdentity.buf[1] = (nrcellidentity & 0x00ff000000)>>24;
		nRCellIdentity.buf[0] = (nrcellidentity & 0xff00000000)>>32;

		return true;
	}
	bool NRCellIdentity::decodefrombitstring(Ngap_NRCellIdentity_t &nRCellIdentity)
	{
		if(!nRCellIdentity.buf) return false;

		nrcellidentity = nRCellIdentity.buf[0];
		nrcellidentity = nrcellidentity<<32;
		nrcellidentity |= nRCellIdentity.buf[1]<<24;
		nrcellidentity |= nRCellIdentity.buf[2]<<16;
		nrcellidentity |= nRCellIdentity.buf[3]<<8;
		nrcellidentity |= nRCellIdentity.buf[4];

		return true;
	}
	unsigned long NRCellIdentity::getNRCellIdentity()
	{
		return nrcellidentity;
	}
}

