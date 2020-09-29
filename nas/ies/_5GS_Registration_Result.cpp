#include "_5GS_Registration_Result.hpp"
#include "logger.hpp"
using namespace nas;

_5GS_Registration_Result::_5GS_Registration_Result(uint8_t iei) {
	_iei = iei;
}
_5GS_Registration_Result::_5GS_Registration_Result(const uint8_t iei, bool emergency, bool nssaa, bool sms, uint8_t value) {
	_iei = iei;
	Emergency = emergency;
	NSSAA = nssaa;
	SMS = sms;
	_value = value;
}
_5GS_Registration_Result::_5GS_Registration_Result() {}
_5GS_Registration_Result::~_5GS_Registration_Result() {}

void _5GS_Registration_Result::setValue(uint8_t value) {
	_value = value;
}

uint8_t _5GS_Registration_Result::getValue() {
	return _value;
}

int _5GS_Registration_Result::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding _5GS_Registration_Result iei(0x%x)", _iei);
	if (len < 3) {
		Logger::nas_mm().error("len is less than 3");
		return 0;
	}
	uint8_t octet=0;
	int encoded_size = 0;
	octet = 0x00 | (Emergency << 5) | (NSSAA << 4) | (SMS << 3) | (_value & 0x07);
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = 1;        encoded_size++;
		*(buf + encoded_size) = octet;           encoded_size++;
	Logger::nas_mm().debug("encoded _5GS_Registration_Result _value(0x%x),iei(0x%x)", *(buf+encoded_size-1),_iei);
	}
	else {
		*(buf + encoded_size) = 1;        encoded_size++;
		*(buf + encoded_size) = octet;           encoded_size++;
	}
	Logger::nas_mm().debug("encoded _5GS_Registration_Result len(%d)", encoded_size);
	return encoded_size;
}

int _5GS_Registration_Result::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding _5GS_Registration_Result iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		_iei=*buf;
		decoded_size++;
	}
	uint8_t octet=0;
//	length = *(buf + decoded_size); 
	decoded_size++;
	octet = *(buf + decoded_size); decoded_size++;
	Emergency = (octet & 0x20)>>5;
	NSSAA = (octet & 0x10) >> 4;
	SMS = (octet & 0x08) >> 3;
	_value = octet & 0x07;
	Logger::nas_mm().debug("decoded _5GS_Registration_Result Emergency(0x%x) NSSAA(0x%x) SMS(0x%x) _value(0x%x)", Emergency, NSSAA, SMS, _value);
	Logger::nas_mm().debug("decoded _5GS_Registration_Result len(%d)", decoded_size);
	return decoded_size;
}



