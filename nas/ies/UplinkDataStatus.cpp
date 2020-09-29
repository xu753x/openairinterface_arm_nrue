#include "UplinkDataStatus.hpp"
#include "logger.hpp"
using namespace nas;

UplinkDataStatus::UplinkDataStatus(uint8_t iei) {
	_iei = iei;
}
UplinkDataStatus::UplinkDataStatus(const uint8_t iei, uint16_t value) {
	_iei = iei;
	_value = value;
	length = 4;
}
UplinkDataStatus::UplinkDataStatus() {}

UplinkDataStatus::~UplinkDataStatus() {}

void UplinkDataStatus::setValue(uint8_t iei, uint16_t value) {
	_iei = iei;
	_value = value;
}
uint16_t UplinkDataStatus::getValue() {
	return _value;
}

int UplinkDataStatus::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding UplinkDataStatus iei(0x%x)", _iei);
	if (len < length) {
		Logger::nas_mm().error("len is less than %d", length);
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = length - 2; encoded_size++;
		*(buf + encoded_size) = (_value&0xff00)>>8; encoded_size++;
 		*(buf + encoded_size) = _value&0x00ff; encoded_size++;
	}
	else {
//		*(buf + encoded_size) = length - 1; encoded_size++;
//		*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded UplinkDataStatus len(%d)", encoded_size);
	return encoded_size;
}

int UplinkDataStatus::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding UplinkDataStatus iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	length = *(buf + decoded_size); decoded_size++;
	_value = *(buf + decoded_size)<<8; decoded_size++;
	_value |= *(buf + decoded_size);  decoded_size++;
	Logger::nas_mm().debug("decoded UplinkDataStatus value(0x%4x)",_value);
	Logger::nas_mm().debug("decoded UplinkDataStatus len(%d)", decoded_size);
	return decoded_size;
}

