#include "AuthenticationResponse.hpp"
#include "3gpp_ts24501.hpp"
#include "logger.hpp"

using namespace nas;

AuthenticationResponse::AuthenticationResponse() {
	Logger::nas_mm().debug("initiating class AuthenticationResponse");
	plain_header = NULL;
	ie_authentication_response_parameter = NULL;
	ie_eap_message = NULL;
}

AuthenticationResponse::~AuthenticationResponse() {}

void AuthenticationResponse::setHeader(uint8_t security_header_type) {
	plain_header = new NasMmPlainHeader();
	plain_header->setHeader(EPD_5GS_MM_MSG, security_header_type, AUTHENTICATION_RESPONSE);
}

void AuthenticationResponse::setAuthentication_Response_Parameter(bstring para) {
	ie_authentication_response_parameter = new Authentication_Response_Parameter(0x2D, para);
}
bool AuthenticationResponse::getAuthenticationResponseParameter(bstring &para) {
	  if (ie_authentication_response_parameter) {
		   ie_authentication_response_parameter->getValue(para);
		return true;
	  }
	  else { return false; }
  }
void AuthenticationResponse::setEAP_Message(bstring eap) {
	ie_eap_message = new EAP_Message(0x78, eap);
}
bool AuthenticationResponse::getEapMessage(bstring &eap) {
         if (ie_eap_message) {
                     ie_eap_message->getValue(eap);
                  return 0;
            }
            else { return -1; }
    }

int AuthenticationResponse::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding AuthenticationResponse message");
	int encoded_size = 0;
	if (!plain_header) {
		Logger::nas_mm().error("Mandontary IE missing Header");
		return 0;
	}
	if (!(plain_header->encode2buffer(buf, len))) return 0;
	encoded_size += 3;
	if (!ie_authentication_response_parameter) {
		Logger::nas_mm().warn("IE ie_authentication_response_parameter is not avaliable");
	}
	else {
		if (int size = ie_authentication_response_parameter->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_authentication_response_parameter  error");
			return 0;
		}
	}
	if (!ie_eap_message) {
		Logger::nas_mm().warn("IE ie_eap_message is not avaliable");
	}
	else {
		if (int size = ie_eap_message->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_eap_message  error");
			return 0;
		}
	}
	Logger::nas_mm().debug("encoded AuthenticationResponse message len(%d)", encoded_size);
	return 1;
}

int AuthenticationResponse::decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len) {
	Logger::nas_mm().debug("decoding AuthenticationResponse message");
	int decoded_size = 3;
	plain_header = header;
	Logger::nas_mm().debug("decoded_size(%d)", decoded_size);
	uint8_t octet = *(buf + decoded_size);
	Logger::nas_mm().debug("first option iei(0x%x)", octet);
	while ((octet != 0x0)) {
		switch (octet) {
		case 0x2D: {
			Logger::nas_mm().debug("decoding iei(0x2D)");
			ie_authentication_response_parameter = new Authentication_Response_Parameter();
			decoded_size += ie_authentication_response_parameter->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
			octet = *(buf + decoded_size);
			Logger::nas_mm().debug("next iei(0x%x)", octet);
		}break;
		case 0x78: {
			Logger::nas_mm().debug("decoding iei(0x78)");
			ie_eap_message = new EAP_Message();
			decoded_size += ie_eap_message->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
			octet = *(buf + decoded_size);
			Logger::nas_mm().debug("next iei(0x%x)", octet);
		}break;
		}
	}
	Logger::nas_mm().debug("decoded AuthenticationResponse message len(%d)", decoded_size);


}



