#include "Payload_Container_Type.hpp"
#include "logger.hpp"
using namespace nas;

Payload_Container_Type::Payload_Container_Type(const uint8_t iei, uint8_t value) {
	_iei = iei;
	_value = value;
}
Payload_Container_Type::Payload_Container_Type() {}
Payload_Container_Type::~Payload_Container_Type() {};
void Payload_Container_Type::setValue(const uint8_t value) {
	_value = value;
}
uint8_t Payload_Container_Type::getValue() {
	return _value;
}

int Payload_Container_Type::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding Payload_Container_Type IE iei(0x%x)", _iei);
	if (len < 1) {
		Logger::nas_mm().error("len is less than one");
		return -1;
	}
	else {
		uint8_t octet = 0;
		if (!(_iei & 0x0f)) {
			octet =(_value & 0x0f);
			*buf = octet;
			Logger::nas_mm().debug("encoded Payload_Container_Type IE(len(1/2 octet))");
			return 1;
		}
		else {
			octet = (_iei << 4) | (_value&0x0f);
			*buf = octet;
			Logger::nas_mm().debug("encoded Payload_Container_Type IE(len(1 octet))");
			return 1;
		}
	}
}

int Payload_Container_Type::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding Payload_Container_Type IE");
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
		Logger::nas_mm().debug("decoded Payload_Container_Type iei(0x%x) value(0x%x)", _iei,_value);
		return 1;
	}
}





