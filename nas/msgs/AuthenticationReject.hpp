#ifndef _AuthenticationReject_H_
#define _AuthenticationReject_H_

#include "nas_ie_header.hpp"

namespace nas {

	class AuthenticationReject {
	public:
		AuthenticationReject();
		~AuthenticationReject();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);
		void setEAP_Message(bstring eap);

	public:
		NasMmPlainHeader      *plain_header;
		EAP_Message           *ie_eap_message;
	};













}



















#endif




