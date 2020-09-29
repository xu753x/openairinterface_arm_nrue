#include "ULNASTransport.hpp"
#include "3gpp_ts24501.hpp"
#include "logger.hpp"

using namespace nas;

ULNASTransport::ULNASTransport() {
	Logger::nas_mm().debug("initiating class ULNASTransport");
	plain_header = NULL;
	ie_payload_container_type = NULL;
	ie_payload_container = NULL;
        ie_pdu_session_identity_2 = NULL;
        ie_old_pdu_session_identity_2 = NULL;
        ie_request_type = NULL;
        ie_s_nssai = NULL;
        ie_dnn = NULL;
        ie_additional_information = NULL;
        ie_ma_pdu_session_information = NULL;
        ie_release_assistance_indication = NULL;
}

ULNASTransport::~ULNASTransport() {}

void ULNASTransport::setHeader(uint8_t security_header_type) {
	plain_header = new NasMmPlainHeader();
	plain_header->setHeader(EPD_5GS_MM_MSG, security_header_type, UL_NAS_TRANSPORT);
}
void ULNASTransport::setPayload_Container_Type(uint8_t value) {
	ie_payload_container_type = new Payload_Container_Type(0x00, value);
}
uint8_t ULNASTransport::getPayloadContainerType(){
		if (ie_payload_container_type) {
		return 	ie_payload_container_type->getValue();
		}
		else { return -1; }
	}
void ULNASTransport::setPayload_Container(std::vector<PayloadContainerEntry> content) {
	ie_payload_container = new Payload_Container(0x00,content);
}

bool ULNASTransport::getPayloadContainer(bstring & content){
  ie_payload_container->getValue(content); 
  return true;
}

bool ULNASTransport::getPayloadContainer(std::vector<PayloadContainerEntry> &content) {
			if (ie_payload_container) {
				ie_payload_container->getValue(content);
					return 0;
			}
			else { return -1; }
		}
void ULNASTransport::setPDU_Session_Identity_2(uint8_t value) {
	ie_pdu_session_identity_2 = new PDU_Session_Identity_2(0x12, value);
}
uint8_t ULNASTransport::getPduSessionId() {
			if (ie_pdu_session_identity_2) {
				return ie_pdu_session_identity_2->getValue();
			}
			else { return -1; }
		}
void ULNASTransport::setOLD_PDU_Session_Identity_2(uint8_t value) {
	ie_old_pdu_session_identity_2 = new PDU_Session_Identity_2(0x59, value);
}
uint8_t ULNASTransport::getOldPduSessionId() {
			if (ie_old_pdu_session_identity_2) {
				return ie_old_pdu_session_identity_2->getValue();
			}
			else { return -1; }
		}
void ULNASTransport::setRequest_Type(uint8_t value) {
	ie_request_type = new Request_Type(0x08, value);
}
uint8_t ULNASTransport::getRequestType() {
			if (ie_request_type) {
				return ie_request_type->getValue();
			}
			else { return -1; }
		}
void ULNASTransport::setS_NSSAI(SNSSAI_s snssai) {
	ie_s_nssai = new S_NSSAI(0x22, snssai);
}
bool ULNASTransport::getSnssai(SNSSAI_s &snssai) {
			if (ie_s_nssai) {
				ie_s_nssai->getValue(snssai);
				return 0;
			}
			else { return -1; }
		}
void ULNASTransport::setDNN(bstring dnn) {
	ie_dnn = new DNN(0x25,dnn);
}
bool ULNASTransport::getDnn(bstring &dnn) {
			if (ie_dnn) {
				ie_dnn->getValue(dnn);
				return true;
			}
			else { return false; }
		}
void ULNASTransport::setAdditional_Information(uint8_t _length, uint8_t value) {
	ie_additional_information = new Additional_Information(0x24,_length, value);
}
void ULNASTransport::setMA_PDU_Session_Information(uint8_t value) {
	ie_ma_pdu_session_information = new MA_PDU_Session_Information(0x0A, value);
}
void ULNASTransport::setRelease_Assistance_Indication(uint8_t value) {
	ie_release_assistance_indication = new Release_Assistance_Indication(0x0F, value);
}
int ULNASTransport::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding ULNASTransport message");
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
	if (!ie_old_pdu_session_identity_2) {
		Logger::nas_mm().warn("IE ie_old_pdu_session_identity_2 is not avaliable");
	}
	else {
		if (int size = ie_old_pdu_session_identity_2->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie ie_old_pdu_session_identity_2  error");
			return 0;
		}

	}
	if (!ie_request_type) {
		Logger::nas_mm().warn("IE ie_request_type is not avaliable");
	}
	else {
		if (int size = ie_request_type->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie ie_request_type  error");
			return 0;
		}

	}
	if (!ie_s_nssai) {
		Logger::nas_mm().warn("IE ie_s_nssai is not avaliable");
	}
	else {
		if (int size = ie_s_nssai->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie ie_s_nssai  error");
			return 0;
		}

	}
	if (!ie_dnn) {
		Logger::nas_mm().warn("IE ie_dnn is not avaliable");
	}
	else {
		if (int size = ie_dnn->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie ie_dnn  error");
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
	if (!ie_ma_pdu_session_information) {
		Logger::nas_mm().warn("IE ie_ma_pdu_session_information is not avaliable");
	}
	else {
		if (int size = ie_ma_pdu_session_information->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie ie_ma_pdu_session_information  error");
			return 0;
		}

	}
	if (!ie_release_assistance_indication) {
		Logger::nas_mm().warn("IE ie_release_assistance_indication is not avaliable");
	}
	else {
		if (int size = ie_release_assistance_indication->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie ie_release_assistance_indication  error");
			return 0;
		}

	}
	Logger::nas_mm().debug("encoded ULNASTransport message len(%d)", encoded_size);
	return 1;
}

int ULNASTransport::decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len) {
	Logger::nas_mm().debug("decoding ULNASTransport message");
	int decoded_size = 3;
	plain_header = header;
	ie_payload_container_type = new Payload_Container_Type();
	decoded_size += ie_payload_container_type->decodefrombuffer(buf + decoded_size, len - decoded_size, false);
	ie_payload_container = new Payload_Container();
	decoded_size += ie_payload_container->decodefrombuffer(buf + decoded_size, len - decoded_size, false, ie_payload_container_type->getValue());
	Logger::nas_mm().debug("decoded_size(%d)", decoded_size);
	uint8_t octet = *(buf + decoded_size);
	Logger::nas_mm().debug("first option iei(0x%x)", octet);
	while ((octet != 0x0)) {
	switch ((octet & 0xf0) >> 4) {
	case 0x8: {
	Logger::nas_mm().debug("decoding iei(0x8)");
	ie_request_type = new Request_Type();
	decoded_size += ie_request_type->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
	octet = *(buf + decoded_size);
	Logger::nas_mm().debug("next iei(0x%x)", octet);
	}break;	
	case 0xA: {
		Logger::nas_mm().debug("decoding iei(0xA)");
		ie_ma_pdu_session_information = new MA_PDU_Session_Information();
		decoded_size += ie_ma_pdu_session_information->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
		octet = *(buf + decoded_size);
		Logger::nas_mm().debug("next iei(0x%x)", octet);
	}break;
	case 0xF: {
		Logger::nas_mm().debug("decoding iei(0xF)");
		ie_release_assistance_indication = new Release_Assistance_Indication();
		decoded_size += ie_release_assistance_indication->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
		octet = *(buf + decoded_size);
		Logger::nas_mm().debug("next iei(0x%x)", octet);
	}break;
	}
	switch (octet) {
	case 0x12: {
		Logger::nas_mm().debug("decoding iei(0x12)");
		ie_pdu_session_identity_2 = new PDU_Session_Identity_2();
		decoded_size += ie_pdu_session_identity_2->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
		octet = *(buf + decoded_size);
		Logger::nas_mm().debug("next iei(0x%x)", octet);
	}break;
	case 0x59: {
		Logger::nas_mm().debug("decoding iei(0x59)");
		ie_old_pdu_session_identity_2 = new PDU_Session_Identity_2();
		decoded_size += ie_old_pdu_session_identity_2->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
		octet = *(buf + decoded_size);
		Logger::nas_mm().debug("next iei(0x%x)", octet);
	}break;
	case 0x22: {
		Logger::nas_mm().debug("decoding iei(0x22)");
		ie_s_nssai = new S_NSSAI();
		decoded_size += ie_s_nssai->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
		octet = *(buf + decoded_size);
		Logger::nas_mm().debug("next iei(0x%x)", octet);
	}break;
	case 0x25: {
		Logger::nas_mm().debug("decoding iei(0x25)");
		ie_dnn = new DNN();
		decoded_size += ie_dnn->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
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
	}
}
	Logger::nas_mm().debug("decoded ULNASTransport message len(%d)", decoded_size);


}








