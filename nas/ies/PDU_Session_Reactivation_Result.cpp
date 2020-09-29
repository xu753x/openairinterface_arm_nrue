#include "PDU_Session_Reactivation_Result.hpp"
#include "logger.hpp"
using namespace nas;

PDU_Session_Reactivation_Result::PDU_Session_Reactivation_Result(uint8_t iei) {
	_iei = iei;
}
PDU_Session_Reactivation_Result::PDU_Session_Reactivation_Result(const uint8_t iei, uint16_t value) {
	_iei = iei;
	_value = value;
	length = 4;
}
PDU_Session_Reactivation_Result::PDU_Session_Reactivation_Result() {}
PDU_Session_Reactivation_Result::~PDU_Session_Reactivation_Result() {}

void PDU_Session_Reactivation_Result::setValue(uint8_t iei, uint16_t value) {
	_iei = iei;
	_value = value;
}
uint16_t PDU_Session_Reactivation_Result::getValue() {
	return _value;
}

int PDU_Session_Reactivation_Result::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding PDU_Session_Reactivation_Result iei(0x%x)", _iei);
	if (len < length) {
		Logger::nas_mm().error("len is less than %d", length);
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = length - 2; encoded_size++;
		*(buf + encoded_size) = (_value&0x00ff); encoded_size++;
		*(buf + encoded_size) = (_value&0xff00)>>8; encoded_size++;
	}
	else {
		//*(buf + encoded_size) = length - 1; encoded_size++;
		//*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded PDU_Session_Reactivation_Result len(%d)", encoded_size);
	return encoded_size;
}

int PDU_Session_Reactivation_Result::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding PDU_Session_Reactivation_Result iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	_value = 0x0000;
	length = *(buf + decoded_size); decoded_size++;
	_value |= *(buf + decoded_size); decoded_size++;
	_value |= (*(buf + decoded_size))<<8; decoded_size++;
	Logger::nas_mm().debug("decoded PDU_Session_Reactivation_Result value(0x%4x)", _value);
	Logger::nas_mm().debug("decoded PDU_Session_Reactivation_Result len(%d)", decoded_size);
	return decoded_size;
}



