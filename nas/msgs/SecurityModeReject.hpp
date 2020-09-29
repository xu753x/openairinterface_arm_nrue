#ifndef _SecurityModeReject_H_
#define _SecurityModeReject_H_

#include "nas_ie_header.hpp"

namespace nas {

	class SecurityModeReject {
	public:
		SecurityModeReject();
		~SecurityModeReject();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);
		void set_5GMM_Cause(uint8_t value);
	public:
		NasMmPlainHeader      *plain_header;
		_5GMM_Cause            *ie_5gmm_cause;
	};













}



















#endif





