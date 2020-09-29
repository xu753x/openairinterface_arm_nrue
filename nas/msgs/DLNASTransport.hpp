#ifndef _DLNASTransport_H_
#define _DLNASTransport_H_

#include "nas_ie_header.hpp"

namespace nas {

	class DLNASTransport {
	public:
		DLNASTransport();
		~DLNASTransport();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);
		void setPayload_Container_Type(uint8_t value);
		void setPayload_Container(std::vector<PayloadContainerEntry> content);
                void setPayload_Container(uint8_t *buf, int len);
		void setPDUSessionId(uint8_t value);
		void setAdditional_Information(uint8_t _length, uint8_t value);
		void set_5GMM_Cause(uint8_t value);
		void setBack_off_timer_value(uint8_t unit, uint8_t value);
	public:
		NasMmPlainHeader      *plain_header;
		Payload_Container_Type *ie_payload_container_type;
		Payload_Container  *ie_payload_container;
		PDU_Session_Identity_2 *ie_pdu_session_identity_2;
		Additional_Information *ie_additional_information;
		_5GMM_Cause            *ie_5gmm_cause;
		GPRS_Timer_3           *ie_back_off_timer_value;
	};













}



















#endif









