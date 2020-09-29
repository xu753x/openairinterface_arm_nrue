#include "AuthenticationFailure.hpp"
#include "3gpp_ts24501.hpp"
#include "logger.hpp"

using namespace nas;

AuthenticationFailure::AuthenticationFailure() {
	Logger::nas_mm().debug("initiating class AuthenticationFailure");
	plain_header = NULL;
	ie_5gmm_cause = NULL;
	ie_authentication_failure_parameter = NULL;
}

AuthenticationFailure::~AuthenticationFailure() {}

void AuthenticationFailure::setHeader(uint8_t security_header_type) {
	plain_header = new NasMmPlainHeader();
	plain_header->setHeader(EPD_5GS_MM_MSG, security_header_type, AUTHENTICATION_FAILURE);
}
void AuthenticationFailure::set_5GMM_Cause(uint8_t value) {
	ie_5gmm_cause = new _5GMM_Cause(0x00,value);
}
uint8_t AuthenticationFailure::get5GMmCause() {
			if (ie_5gmm_cause) {
				return ie_5gmm_cause->getValue();
			}
			else { return -1; }
		}
void AuthenticationFailure::setAuthentication_Failure_Parameter(bstring auts) {
	ie_authentication_failure_parameter = new Authentication_Failure_Parameter(0x30, auts);
}
bool AuthenticationFailure::getAutsInAuthFailPara(bstring &auts) {
			if (ie_authentication_failure_parameter) {
				ie_authentication_failure_parameter->getValue(auts);
				return true;
			}
			else { return false; }
		}

int AuthenticationFailure::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding AuthenticationFailure message");
	int encoded_size = 0;
	if (!plain_header) {
		Logger::nas_mm().error("Mandontary IE missing Header");
		return 0;
	}
	if (!(plain_header->encode2buffer(buf, len))) return 0;
	encoded_size += 3;
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
	if (!ie_authentication_failure_parameter) {
		Logger::nas_mm().warn("IE ie_authentication_failure_parameter is not avaliable");
	}
	else {
		if (int size = ie_authentication_failure_parameter->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_authentication_failure_parameter  error");
			return 0;
		}
	}
	
	Logger::nas_mm().debug("encoded AuthenticationFailure message len(%d)", encoded_size);
	return 1;
}

int AuthenticationFailure::decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len) {
	Logger::nas_mm().debug("decoding AuthenticationFailure message");
	int decoded_size = 3;
	plain_header = header;
	ie_5gmm_cause = new _5GMM_Cause();
	decoded_size += ie_5gmm_cause->decodefrombuffer(buf + decoded_size, len - decoded_size, false);
	Logger::nas_mm().debug("decoded_size(%d)", decoded_size);
	uint8_t octet = *(buf + decoded_size);
	Logger::nas_mm().debug("first option iei(0x%x)", octet);
	while ((octet != 0x0)) {
		switch (octet) {
		case 0x30: {
			Logger::nas_mm().debug("decoding iei(0x30)");
			ie_authentication_failure_parameter = new Authentication_Failure_Parameter();
			decoded_size += ie_authentication_failure_parameter->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
			octet = *(buf + decoded_size);
			Logger::nas_mm().debug("next iei(0x%x)", octet);
		}break;
		}
	}
	Logger::nas_mm().debug("decoded AuthenticationFailure message len(%d)", decoded_size);


}




