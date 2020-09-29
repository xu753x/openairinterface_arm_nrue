#include "PDU_Session_Identity_2.hpp"
#include "logger.hpp"
using namespace nas;

PDU_Session_Identity_2::PDU_Session_Identity_2(uint8_t iei) {
	_iei = iei;
}
PDU_Session_Identity_2::PDU_Session_Identity_2(const uint8_t iei, uint8_t value) {
	_iei = iei;
	_value = value;
}
PDU_Session_Identity_2::PDU_Session_Identity_2() {}
PDU_Session_Identity_2::~PDU_Session_Identity_2() {}

void PDU_Session_Identity_2::setValue(uint8_t value) {
	_value = value;
}

uint8_t PDU_Session_Identity_2::getValue() {
	return _value;
}

int PDU_Session_Identity_2::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding PDU_Session_Identity_2 iei(0x%x)", _iei);
	if (len < 2) {
		Logger::nas_mm().error("len is less than 2");
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = _value;           encoded_size++;
	}
	else {
	//	*(buf + encoded_size) = length - 1; encoded_size++;
	//	*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded PDU_Session_Identity_2 len(%d)", encoded_size);
	return encoded_size;
}

int PDU_Session_Identity_2::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding PDU_Session_Identity_2 iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	_value=0x00;
	_value = *(buf + decoded_size); decoded_size++;
	Logger::nas_mm().debug("decoded PDU_Session_Identity_2 value(0x%x)", _value);
	Logger::nas_mm().debug("decoded PDU_Session_Identity_2 len(%d)", decoded_size);
	return decoded_size;
}



