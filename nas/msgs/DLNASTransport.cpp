#include "DLNASTransport.hpp"
#include "3gpp_ts24501.hpp"
#include "logger.hpp"
#include "bstrlib.h"

using namespace nas;

DLNASTransport::DLNASTransport() {
	Logger::nas_mm().debug("initiating class DLNASTransport");
	plain_header = NULL;
	ie_payload_container_type = NULL;
	ie_payload_container = NULL;
	ie_pdu_session_identity_2 = NULL;
	ie_additional_information = NULL;
	ie_5gmm_cause = NULL;
	ie_back_off_timer_value = NULL;
}

DLNASTransport::~DLNASTransport() {}

void DLNASTransport::setHeader(uint8_t security_header_type) {
	plain_header = new NasMmPlainHeader();
	plain_header->setHeader(EPD_5GS_MM_MSG, security_header_type, DL_NAS_TRANSPORT);
}
void DLNASTransport::setPayload_Container_Type(uint8_t value) {
	ie_payload_container_type = new Payload_Container_Type(0x00, value);
}
void DLNASTransport::setPayload_Container(std::vector<PayloadContainerEntry> content) {
	ie_payload_container = new Payload_Container(0x00,content);
}

void DLNASTransport::setPayload_Container(uint8_t *buf, int len){
        bstring b = blk2bstr(buf, len);
        ie_payload_container = new Payload_Container(0x00, b);
}

void DLNASTransport::setPDUSessionId(uint8_t value) {
	ie_pdu_session_identity_2 = new PDU_Session_Identity_2(0x12, value);
}
void DLNASTransport::setAdditional_Information(uint8_t _length, uint8_t value) {
	ie_additional_information = new Additional_Information(0x24,_length, value);
}
void DLNASTransport::set_5GMM_Cause(uint8_t value) {
	ie_5gmm_cause = new _5GMM_Cause(0x58,value);
}
void DLNASTransport::setBack_off_timer_value(uint8_t unit, uint8_t value) {
	ie_back_off_timer_value = new GPRS_Timer_3(0x37, unit, value);
}
int DLNASTransport::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding DLNASTransport message");
	int encoded_size = 0;
	if (!plain_header) {
		Logger::nas_mm().error("Mandontary IE missing Header");
		return 0;
	}
	if (!(plain_header->encode2buffer(buf, len))) return 0;
	encoded_size += 3;
	if (!ie_payload_container_type) {
		Logger::nas_mm().warn("IE ie_payload_container_type is not avaliable");
	}
	else {
		if (int size = ie_payload_container_type->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_payload_container_type  error");
			return 0;
		}
	}
	if (!ie_payload_container) {
		Logger::nas_mm().warn("IE ie_payload_container is not avaliable");
	}
	else {
		if (int size = ie_payload_container->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_payload_container  error");
			return 0;
		}
	}
	if (!ie_pdu_session_identity_2) {
		Logger::nas_mm().warn("IE ie_pdu_session_identity_2 is not avaliable");
	}
	else {
		if (int size = ie_pdu_session_identity_2->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie ie_pdu_session_identity_2  error");
			return 0;
		}

	}
	if (!ie_additional_information) {
		Logger::nas_mm().warn("IE ie_additional_information is not avaliable");
	}
	else {
		if (int size = ie_additional_information->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie ie_additional_information  error");
			return 0;
		}

	}
	if (!ie_5gmm_cause) {
		Logger::nas_mm().warn("IE ie_5gmm_cause is not avaliable");
	}
	else {
		if (int size = ie_5gmm_cause->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_5gmm_cause  error");
		}
	}
	if (!ie_back_off_timer_value) {
		Logger::nas_mm().warn("IE ie_back_off_timer_value is not avaliable");
	}
	else {
		if (int size = ie_back_off_timer_value->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_back_off_timer_value  error");
			return 0;
		}
	}
	Logger::nas_mm().debug("encoded DLNASTransport message len(%d)", encoded_size);
	return encoded_size;
}

int DLNASTransport::decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len) {
	Logger::nas_mm().debug("decoding DLNASTransport message");
	int decoded_size = 3;
	plain_header = header;
	ie_payload_container_type = new Payload_Container_Type();
	decoded_size += ie_payload_container_type->decodefrombuffer(buf + decoded_size, len - decoded_size, false);
	ie_payload_container = new Payload_Container();
	decoded_size += ie_payload_container->decodefrombuffer(buf + decoded_size, len - decoded_size, false);
	Logger::nas_mm().debug("decoded_size(%d)", decoded_size);
	uint8_t octet = *(buf + decoded_size);
	Logger::nas_mm().debug("first option iei(0x%x)", octet);
	while ((octet != 0x0)) {
	switch (octet) {
	case 0x12: {
		Logger::nas_mm().debug("decoding iei(0x12)");
		ie_pdu_session_identity_2 = new PDU_Session_Identity_2();
		decoded_size += ie_pdu_session_identity_2->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
		octet = *(buf + decoded_size);
		Logger::nas_mm().debug("next iei(0x%x)", octet);
	}break;
	case 0x24: {
		Logger::nas_mm().debug("decoding iei(0x24)");
		ie_additional_information = new Additional_Information();
		decoded_size += ie_additional_information->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
		octet = *(buf + decoded_size);
		Logger::nas_mm().debug("next iei(0x%x)", octet);
	}break;
	case 0x58: {
		Logger::nas_mm().debug("decoding iei(0x58)");
		ie_5gmm_cause = new _5GMM_Cause();
		decoded_size += ie_5gmm_cause->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
		octet = *(buf + decoded_size);
		Logger::nas_mm().debug("next iei(0x%x)", octet);
	}break;
	case 0x37: {
		Logger::nas_mm().debug("decoding iei(0x37)");
		ie_back_off_timer_value = new GPRS_Timer_3();
		decoded_size += ie_back_off_timer_value->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
		octet = *(buf + decoded_size);
		Logger::nas_mm().debug("next iei(0x%x)", octet);
	}break;
	}
}
	Logger::nas_mm().debug("decoded DLNASTransport message len(%d)", decoded_size);


}









