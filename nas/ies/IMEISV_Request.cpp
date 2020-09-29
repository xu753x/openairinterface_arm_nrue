#include "IMEISV_Request.hpp"
#include "logger.hpp"
using namespace nas;

IMEISV_Request::IMEISV_Request(uint8_t iei) {
	_iei = iei;
}
IMEISV_Request::IMEISV_Request(const uint8_t iei,uint8_t value) {
	_iei = iei;
	_value = value;
}
IMEISV_Request::IMEISV_Request() {}
IMEISV_Request::~IMEISV_Request() {}

void IMEISV_Request::setValue(uint8_t value) {
	_value = value;
}


uint8_t IMEISV_Request::getValue() {
	return _value;
}

int IMEISV_Request::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding IMEISV_Request iei(0x%x)", _iei);
	if (len < 1) {
		Logger::nas_mm().error("len is less than 1");
		return 0;
	}
	uint8_t octet = 0;
	if (!(_iei & 0x0f)) {
		//octet = (0x0f) & ((tsc << 3) | key_id);
		//*buf = octet;
		//Logger::nas_mm().debug("encoded Payload_Container_Type IE(len(1/2 octet))");
		//return 0;
	}
	else {
		octet = (_iei << 4) | (_value&0x07);
		*buf = octet;
		Logger::nas_mm().debug("encoded IMEISV_Request IE(len(1 octet))");
		return 1;
	}
}

int IMEISV_Request::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	if (len < 1) {
	Logger::nas_mm().error("len is less than one");
	return 0;
	}
	else {
	uint8_t octet = (*buf);
	if (is_option) {
		_iei = (octet & 0xf0) >> 4;
	}
	else {
		_iei = 0;
	}
	_value =octet& 0x07; 
	Logger::nas_mm().debug("decoded IMEISV_Request value(0x%x)", _value);
	return 1;
}
}



