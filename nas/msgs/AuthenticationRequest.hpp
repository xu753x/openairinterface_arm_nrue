#ifndef _AuthenticationRequest_H_
#define _AuthenticationRequest_H_

#include "nas_ie_header.hpp"

namespace nas {

	class AuthenticationRequest {
	public:
		AuthenticationRequest();
		~AuthenticationRequest();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);
		void setngKSI(uint8_t tsc, uint8_t key_set_id);
		void setEAP_Message(bstring eap);
		void setABBA(uint8_t length, uint8_t *value);
		void setAuthentication_Parameter_RAND(uint8_t *value);
		void setAuthentication_Parameter_AUTN(uint8_t *value);
	public:
		NasMmPlainHeader      *plain_header;
		NasKeySetIdentifier    *ie_ngKSI;
		ABBA                  *ie_abba;
		Authentication_Parameter_RAND          *ie_authentication_parameter_rand;
		Authentication_Parameter_AUTN           *ie_authentication_parameter_autn;
		EAP_Message           *ie_eap_message;
	};  













}



















#endif


