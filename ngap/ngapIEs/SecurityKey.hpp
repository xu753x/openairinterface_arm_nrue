#ifndef _SECURITYKEY_H_
#define _SECURITYKEY_H_

extern "C"{
  #include "Ngap_SecurityKey.h"
}

namespace ngap{

	class SecurityKey{
	public:
		SecurityKey();
		virtual ~SecurityKey();

		bool encode2bitstring(Ngap_SecurityKey_t &);
		bool decodefrombitstring(Ngap_SecurityKey_t &);
		bool getSecurityKey(uint8_t* &buffer);
		void setSecurityKey(uint8_t* buffer);
	private:
		uint8_t *securitykeybuffer;
		size_t buffersize;
	};

}

#endif
