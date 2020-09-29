#include "RegistrationComplete.hpp"
#include "3gpp_ts24501.hpp"
#include "logger.hpp"

using namespace nas;

RegistrationComplete::RegistrationComplete() {
	Logger::nas_mm().debug("initiating class RegistrationComplete");
	plain_header = NULL;
	ie_sor_transparent_container = NULL;

}

RegistrationComplete::~RegistrationComplete() {}

void RegistrationComplete::setHeader(uint8_t security_header_type) {
	plain_header = new NasMmPlainHeader();
	plain_header->setHeader(EPD_5GS_MM_MSG, security_header_type, REGISTRATION_COMPLETE);
}
void RegistrationComplete::setSOR_Transparent_Container(uint8_t header,uint8_t *value) {
	ie_sor_transparent_container = new SOR_Transparent_Container(0x73,header, value);
}
int RegistrationComplete::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding RegistrationComplete message");
	int encoded_size = 0;
	if (!plain_header) {
		Logger::nas_mm().error("Mandontary IE missing Header");
		return 0;
	}
	if (!(plain_header->encode2buffer(buf, len))) return 0;
	encoded_size += 3;
	if (!ie_sor_transparent_container) {
		Logger::nas_mm().warn("IE ie_sor_transparent_container is not avaliable");
	}
	else {
		if (int size = ie_sor_transparent_container->encode2buffer(buf + encoded_size, len - encoded_size)) {
			encoded_size += size;
		}
		else {
			Logger::nas_mm().error("encoding ie_sor_transparent_container  error");
			return 0;
		}
	}
	Logger::nas_mm().debug("encoded RegistrationComplete message len(%d)", encoded_size);
	return 1;
}

int RegistrationComplete::decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len) {
	Logger::nas_mm().debug("decoding RegistrationComplete message");
	int decoded_size = 3;
	plain_header = header;
	Logger::nas_mm().debug("decoded_size(%d)", decoded_size);
	uint8_t octet = *(buf + decoded_size);
	Logger::nas_mm().debug("first option iei(0x%x)", octet);
	while ((octet != 0x0)) {
		switch (octet) {
		case 0x73: {
			Logger::nas_mm().debug("decoding iei(0x73)");
			ie_sor_transparent_container = new SOR_Transparent_Container();
			decoded_size += ie_sor_transparent_container->decodefrombuffer(buf + decoded_size, len - decoded_size, true);
			octet = *(buf + decoded_size);
			Logger::nas_mm().debug("next iei(0x%x)", octet);		
		}break;
		}
	}
	Logger::nas_mm().debug("decoded RegistrationComplete message len(%d)", decoded_size);


}






