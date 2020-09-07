#include "NAS-PDU.hpp"

namespace ngap{

	NAS_PDU::NAS_PDU()
	{
		naspdubuffer = NULL;
		buffersize = -1;
	}
	NAS_PDU::~NAS_PDU(){}

	
	bool NAS_PDU::encode2octetstring(Ngap_NAS_PDU_t & m_naspdu)
	{
		int ret;
		ret = OCTET_STRING_fromBuf(&m_naspdu,naspdubuffer,buffersize);
		if(ret != 0)return false;
		return true;
	}
	bool NAS_PDU::decodefromoctetstring(Ngap_NAS_PDU_t &m_naspdu)
	{
		naspdubuffer = (char *)m_naspdu.buf;
		buffersize = m_naspdu.size;
		return true;
	}
	bool NAS_PDU::getNasPdu(uint8_t* &buffer,size_t &size)
	{
		buffer = (uint8_t *)naspdubuffer;
		size = buffersize;
		if(!naspdubuffer) return false;
		if(buffersize < 0) return false;

		return true;
	}
	void NAS_PDU::setNasPdu(uint8_t* buffer,size_t size)
	{
		naspdubuffer = (char *)buffer;
		buffersize = size;
	}
}
