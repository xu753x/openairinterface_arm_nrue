#ifndef _AuthenticationResult_H_
#define _AuthenticationResult_H_

#include "nas_ie_header.hpp"

namespace nas {

	class AuthenticationResult {
	public:
		AuthenticationResult();
		~AuthenticationResult();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);
		void setngKSI(uint8_t tsc, uint8_t key_set_id);
		void setEAP_Message(bstring eap);
		void setABBA(uint8_t length, uint8_t *value);
	public:
		NasMmPlainHeader      *plain_header;
		NasKeySetIdentifier    *ie_ngKSI;
		EAP_Message           *ie_eap_message;
		ABBA                  *ie_abba;
	};













}



















#endif



