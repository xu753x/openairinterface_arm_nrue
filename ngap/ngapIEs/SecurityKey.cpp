#include "SecurityKey.hpp"

namespace ngap{

	SecurityKey::SecurityKey()
	{
		securitykeybuffer = NULL;
	}
	SecurityKey::~SecurityKey(){}

	
	bool SecurityKey::encode2bitstring(Ngap_SecurityKey_t &m_securitykey)
	{
		m_securitykey.bits_unused = 0;
		m_securitykey.size = 32;
		uint8_t *buffer = (uint8_t *)calloc(1,32);
		if(!buffer) return false;
		memcpy(buffer,securitykeybuffer,32);
		m_securitykey.buf=buffer;
		
		return true;
	}
	bool SecurityKey::decodefrombitstring(Ngap_SecurityKey_t &m_securitykey)
	{
		securitykeybuffer = m_securitykey.buf;
		return true;
	}
	bool SecurityKey::getSecurityKey(uint8_t* &buffer)
	{
		buffer = (uint8_t *)securitykeybuffer;
		if(!securitykeybuffer) return false;

		return true;
	}
	void SecurityKey::setSecurityKey(uint8_t* buffer)
	{
		securitykeybuffer = buffer;
	}
}
