#ifndef _RegistrationReject_H_
#define _RegistrationReject_H_

#include "nas_ie_header.hpp"

namespace nas {

	class RegistrationReject {
	public:
		RegistrationReject();
		~RegistrationReject();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);
		void set_5GMM_Cause(uint8_t value);
		void setGPRS_Timer_2_3346(uint8_t value);		
		void setGPRS_Timer_2_3502(uint8_t value);
		void setEAP_Message(bstring eap);
		void setRejected_NSSAI(uint8_t cause, uint8_t value);
	public:
		NasMmPlainHeader      *plain_header;
		_5GMM_Cause            *ie_5gmm_cause;
		GPRS_Timer_2          *ie_T3346_value;
		GPRS_Timer_2          *ie_T3502_value;
		EAP_Message           *ie_eap_message;
		Rejected_NSSAI        *ie_rejected_nssai;
	};













}



















#endif

