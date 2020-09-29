#ifndef _IdentityRequest_H_
#define _IdentityRequest_H_

#include "nas_ie_header.hpp"

namespace nas {

	class IdentityRequest {
	public:
		IdentityRequest();
		~IdentityRequest();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);
		void set_5GS_Identity_Type(uint8_t value);
	public:
		NasMmPlainHeader      *plain_header;
		_5GS_Identity_Type    *_5gs_identity_type;
	};













}



















#endif







