#include "_5GS_Identity_Type.hpp"
#include "logger.hpp"
using namespace nas;

_5GS_Identity_Type::_5GS_Identity_Type(uint8_t iei) {
	_iei = iei;
}
_5GS_Identity_Type::_5GS_Identity_Type(const uint8_t iei, uint8_t value) {
	_iei = iei;
	_value = value;
}
_5GS_Identity_Type::_5GS_Identity_Type() {}
_5GS_Identity_Type::~_5GS_Identity_Type() {}

void _5GS_Identity_Type::setValue(uint8_t value) {
	_value = value;
}

uint8_t _5GS_Identity_Type::getValue() {
	return _value;
}

int _5GS_Identity_Type::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding _5GS_Identity_Type iei(0x%x)", _iei);
	if (len < 1) {
		Logger::nas_mm().error("len is less than 1");
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _value&0x07;           encoded_size++;
	}
	else {
		*(buf + encoded_size) = _value & 0x07;           encoded_size++;
	}
	Logger::nas_mm().debug("encoded _5GS_Identity_Type len(%d)", encoded_size);
	return encoded_size;
}

int _5GS_Identity_Type::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding _5GS_Identity_Type iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		_iei=*buf;
		decoded_size++;
	}
	_value=0x00;
//	length = *(buf + decoded_size); 
	_value = *(buf + decoded_size)&0x07; decoded_size++;
	Logger::nas_mm().debug("decoded _5GS_Identity_Type _value(0x%x),iei(0x%x)", _value,_iei);
	Logger::nas_mm().debug("decoded _5GS_Identity_Type len(%d)", decoded_size);
	return decoded_size;
}



