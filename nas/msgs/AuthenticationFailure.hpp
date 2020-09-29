#ifndef _AuthenticationFailure_H_
#define _AuthenticationFailure_H_

#include "nas_ie_header.hpp"

namespace nas {

	class AuthenticationFailure {
	public:
		AuthenticationFailure();
		~AuthenticationFailure();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);
		void set_5GMM_Cause(uint8_t value);
		void setAuthentication_Failure_Parameter(bstring auts);
		
		uint8_t get5GMmCause();
		bool getAutsInAuthFailPara(bstring &auts);
	public:
		NasMmPlainHeader      *plain_header;
		_5GMM_Cause            *ie_5gmm_cause;
		Authentication_Failure_Parameter     *ie_authentication_failure_parameter;
	};













}



















#endif




