#ifndef _AuthenticationResponse_H_
#define _AuthenticationResponse_H_

#include "nas_ie_header.hpp"

namespace nas {

	class AuthenticationResponse {
	public:
		AuthenticationResponse();
		~AuthenticationResponse();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);
		void setAuthentication_Response_Parameter(bstring para);
		void setEAP_Message(bstring eap);
		bool getAuthenticationResponseParameter(bstring &para);
		bool getEapMessage(bstring &eap);
		
	public:
		NasMmPlainHeader      *plain_header;
		Authentication_Response_Parameter    *ie_authentication_response_parameter;
		EAP_Message           *ie_eap_message;
	};













}



















#endif



