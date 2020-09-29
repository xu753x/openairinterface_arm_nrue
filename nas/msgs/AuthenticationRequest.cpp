#include "AuthenticationRequest.hpp"
#include "3gpp_ts24501.hpp"
#include "logger.hpp"

using namespace nas;

AuthenticationRequest::AuthenticationRequest() {
	Logger::nas_mm().debug("initiating class AuthenticationRequest");
	plain_header = NULL;
	ie_ngKSI = NULL;
	ie_abba = NULL;
	ie_authentication_parameter_rand = NULL;
	ie_authentication_parameter_autn = NULL;
	ie_eap_message = NULL;
}

AuthenticationRequest::~AuthenticationRequest() {}

void AuthenticationRequest::setHeader(uint8_t security_header_type) {
	plain_header = new NasMmPlainHeader();
	plain_header->setHeader(EPD_5GS_MM_MSG, security_header_type, AUTHENTICATION_REQUEST);
}

void AuthenticationRequest::setngKSI(uint8_t tsc, uint8_t key_set_id){
	ie_ngKSI = new NasKeySetIdentifier(0x00,tsc, key_set_id);
}
void AuthenticationRequest::setABBA(uint8_t length, uint8_t *value) {
	ie_abba = new ABBA(0x00,length, value);
}
void AuthenticationRequest::setAuthentication_Parameter_RAND(uint8_t *value) {
	ie_authentication_parameter_rand = new Authentication_Parameter_RAND(0x21, value);
}
void AuthenticationRequest::setAuthentication_Parameter_AUTN(uint8_t *value) {
	ie_authentication_parameter_autn = new Authentication_Parameter_AUTN(0x20, value);
}
void AuthenticationRequest::setEAP_Message(bstring eap) {
	ie_eap_message = new EAP_Message(0x78,eap);
}

int AuthenticationRequest::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding AuthenticationRequest message");
	int encoded_size = 0;
	if (!plain_header) {
		Logger::nas_mm().error("Mandontary IE missing Header");
		return 0;
	}
	if (!(plain_header->encode2buffer(buf, len))) return 0;
	encoded_size += 3;
	if (!ie_ngKSI) {
		Logger::nas_mm().warn("IE ie_ngKSI is not avaliable");
	}
	else {
                int size = ie_ngKSI->encode2buffer(buf + encoded_size, len - encoded_size);
		if (size != -1) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_ngKSI  error");
			return 0;
		}
	}
	if (!ie_abba) {
		Logger::nas_mm().warn("IE ie_abba is not avaliable");
	}
	else {
                int size = ie_abba->encode2buffer(buf + encoded_size, len - encoded_size);
		if (size != 0) {
                  Logger::nas_mm().debug("0x%x, 0x%x, 0x%x", (buf+encoded_size)[0],(buf+encoded_size)[1],(buf+encoded_size)[2]);
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_abba  error");
			return 0;
		}
	}
	if (!ie_authentication_parameter_rand) {
		Logger::nas_mm().warn("IE ie_authentication_parameter_rand is not avaliable");
	}
	else {
                int size = ie_authentication_parameter_rand->encode2buffer(buf + encoded_size, len - encoded_size);
		if (size != 0) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_authentication_parameter_rand  error");
			return 0;
		}
	}
	if (!ie_authentication_parameter_autn) {
		Logger::nas_mm().warn("IE ie_authentication_parameter_autn is not avaliable");
	}
	else {
                int size = ie_authentication_parameter_autn->encode2buffer(buf + encoded_size, len - encoded_size);
		if (size != 0) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_authentication_parameter_autn  error");
			return 0;
		}
	}
	if (!ie_eap_message) {
		Logger::nas_mm().warn("IE ie_eap_message is not avaliable");
	}
	else {
                int size = ie_eap_message->encode2buffer(buf + encoded_size, len - encoded_size);
		if (size != 0) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_eap_message  error");
			return 0;
		}
	}
	
	Logger::nas_mm().debug("encoded AuthenticationRequest message len(%d)", encoded_size);
	return encoded_size;
}

int AuthenticationRequest::decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len) {
	Logger::nas_mm().debug("decoding RegistrationReject message");
	int decoded_size = 3;
	plain_header = header;
	ie_ngKSI = new NasKeySetIdentifier();
	decoded_size += ie_ngKSI->decodefrombuffer(buf + decoded_size, len - decoded_size, false, false);
        decoded_size ++;
	ie_abba = new ABBA();
	decoded_size += ie_abba->decodefrombuffer(buf + decoded_size, len - decoded_size, false);
	Logger::nas_mm().debug("decoded_size(%d)", decoded_size);
	uint8_t octet = *(buf + decoded_size);
	Logger::nas_mm().debug("first option iei(0x%x)", octet);
	while ((octet != 0x0)) {
		switch (octet) {
		case 0x21: {
		Logger::nas_mm().debug("decoding iei(0x21)");
			ie_authentication_parameter_rand = new Authentication_Parameter_RAND();
			decoded_size += ie_authentication_parameter_rand->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
			octet = *(buf + decoded_size);
			Logger::nas_mm().debug("next iei(0x%x)", octet);
		}break;
		case 0x20: {
		ie_authentication_parameter_autn = new Authentication_Parameter_AUTN();
			decoded_size += ie_authentication_parameter_autn->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
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
	Logger::nas_mm().debug("decoded AuthenticationRequest message len(%d)", decoded_size);


}


