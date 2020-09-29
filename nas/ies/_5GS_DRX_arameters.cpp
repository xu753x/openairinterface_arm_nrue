#include "_5GS_DRX_arameters.hpp"
#include "logger.hpp"
using namespace nas;

_5GS_DRX_arameters::_5GS_DRX_arameters(uint8_t iei) {
	_iei = iei;
}
_5GS_DRX_arameters::_5GS_DRX_arameters(const uint8_t iei, uint8_t value) {
	_iei = iei;
	_value = value&0x0F;
}
_5GS_DRX_arameters::_5GS_DRX_arameters() {}
_5GS_DRX_arameters::~_5GS_DRX_arameters() {}

void _5GS_DRX_arameters::setValue(uint8_t value) {
	_value = value&0x0F;
}

uint8_t _5GS_DRX_arameters::getValue() {
	return _value;
}

int _5GS_DRX_arameters::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding _5GS_DRX_arameters iei(0x%x)", _iei);
	if (len < 3) {
	//	Logger::nas_mm().error("len is less than %d", length);
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = 1;        encoded_size++;
		*(buf + encoded_size) = 0x0F& _value;           encoded_size++;
	}
	else {
	//	*(buf + encoded_size) = length - 1; encoded_size++;
	//	*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded _5GS_DRX_arameters len(%d)", encoded_size);
	return encoded_size;
}

int _5GS_DRX_arameters::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding _5GS_DRX_arameters iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	_value=0x00;
//	length = *(buf + decoded_size); 
	decoded_size++;
	_value = *(buf + decoded_size) & 0x0f; decoded_size++;
	Logger::nas_mm().debug("decoded _5GS_DRX_arameters DRX value(0x%x)", _value);
	Logger::nas_mm().debug("decoded UE_Status len(%d)", decoded_size);
	return decoded_size;
}


