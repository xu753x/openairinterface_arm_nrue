#include "NSSAI_Inclusion_Mode.hpp"
#include "logger.hpp"
using namespace nas;

NSSAI_Inclusion_Mode::NSSAI_Inclusion_Mode(const uint8_t iei, uint8_t value) {
	_iei = iei;
	_value = value;
}
NSSAI_Inclusion_Mode::NSSAI_Inclusion_Mode() {}
NSSAI_Inclusion_Mode::~NSSAI_Inclusion_Mode() {};
void NSSAI_Inclusion_Mode::setValue(const uint8_t value) {
	_value = value;
}
uint8_t NSSAI_Inclusion_Mode::getValue() {
	return _value;
}

int NSSAI_Inclusion_Mode::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding NSSAI_Inclusion_Mode IE iei(0x%x)", _iei);
	if (len < 1) {
		Logger::nas_mm().error("len is less than one");
		return -1;
	}
	else {
		uint8_t octet = 0;
		if (!(_iei & 0x0f)) {
			//octet = (0x0f) & ((tsc << 3) | key_id);
			//*buf = octet;
			//Logger::nas_mm().debug("encoded Payload_Container_Type IE(len(1/2 octet))");
			//return 0;
		}
		else {
			octet = (_iei << 4) | (_value&0x0f);
			*buf = octet;
			Logger::nas_mm().debug("encoded NSSAI_Inclusion_Mode IE(len(1 octet))");
			return 1;
		}
	}
}

int NSSAI_Inclusion_Mode::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding NSSAI_Inclusion_Mode IE");
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
		_value= octet & 0x0f;
		Logger::nas_mm().debug("decoded NSSAI_Inclusion_Mode iei(0x%x) value(0x%x)", _iei,_value);
		return 1;
	}
}






