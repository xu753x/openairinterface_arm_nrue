#include "UENetworkCapability.hpp"
#include "logger.hpp"
#include <iostream>
using namespace nas;
using namespace std;
UENetworkCapability::UENetworkCapability(uint8_t iei) {
	_iei = iei;
}
UENetworkCapability::UENetworkCapability() {}
UENetworkCapability::~UENetworkCapability() {}

UENetworkCapability::UENetworkCapability(const uint8_t iei, uint8_t _5gg_EEASel, uint8_t _5gg_EIASel)
{
	_iei = iei;
	_5g_EEASel = _5gg_EEASel;
	_5g_EIASel = _5gg_EIASel;
	length = 4;
	Logger::nas_mm().debug("decoded UENetworkCapability EA(0x%d),IA(0x%d)", _5g_EEASel, _5g_EIASel);
}

void UENetworkCapability::setEEASel(uint8_t sel) {
	_5g_EEASel = sel;
}
void UENetworkCapability::setEIASel(uint8_t sel) {
	_5g_EIASel = sel;
}

uint8_t UENetworkCapability::getEEASel() {
	return _5g_EEASel;
}
uint8_t UENetworkCapability::getEIASel() {
	return _5g_EIASel;
}

int UENetworkCapability::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding UENetworkCapability iei(0x%x)", _iei);
	if (len < length) {
		Logger::nas_mm().error("len is less than %d", length);
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = length - 2; encoded_size++;
		*(buf + encoded_size) = _5g_EEASel; encoded_size++;
		*(buf + encoded_size) = _5g_EIASel; encoded_size++;
	}
	else {
		//*(buf + encoded_size) = length - 1; encoded_size++;
		//*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded UENetworkCapability len(%d)", encoded_size);
	return encoded_size;
}

int UENetworkCapability::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding UENetworkCapability iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	length = *(buf + decoded_size); decoded_size++;
	_5g_EEASel = *(buf + decoded_size); decoded_size++;
	_5g_EIASel = *(buf + decoded_size); decoded_size++;
	Logger::nas_mm().debug("decoded UENetworkCapability EA(0x%d),IA(0x%d)", _5g_EEASel, _5g_EIASel);
	Logger::nas_mm().debug("decoded UENetworkCapability len(%d)", decoded_size);
	return decoded_size;
}

