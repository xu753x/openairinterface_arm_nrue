#include "UESecurityCapability.hpp"
#include "logger.hpp"
using namespace nas;

UESecurityCapability::UESecurityCapability(uint8_t iei){
  _iei = iei;
}
UESecurityCapability::UESecurityCapability() {}
UESecurityCapability::~UESecurityCapability(){}

UESecurityCapability::UESecurityCapability(const uint8_t iei, uint8_t _5gg_EASel, uint8_t _5gg_IASel)
{
	_iei = iei;
	_5g_EASel = _5gg_EASel;
	_5g_IASel = _5gg_IASel;
	length = 4;
}

void UESecurityCapability::setEASel(uint8_t sel){
  _5g_EASel = sel;
}
void UESecurityCapability::setIASel(uint8_t sel){
  _5g_IASel = sel;
}

uint8_t UESecurityCapability::getEASel(){
  return _5g_EASel;
}
uint8_t UESecurityCapability::getIASel(){
  return _5g_IASel;
}

int UESecurityCapability::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding UESecurityCapability iei(0x%x)", _iei);
	if (len < length) {
		Logger::nas_mm().error("len is less than %d", length);
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = length - 2; encoded_size++;
		*(buf + encoded_size) = _5g_EASel; encoded_size++; 
		*(buf + encoded_size) = _5g_IASel; encoded_size++; 
	}
	else {
	*(buf + encoded_size) = length - 2; encoded_size++;
		*(buf + encoded_size) = _5g_EASel; encoded_size++;
		*(buf + encoded_size) = _5g_IASel; encoded_size++;
	}
	Logger::nas_mm().debug("encoded UESecurityCapability len(%d)", encoded_size);
	return encoded_size;
}

int UESecurityCapability::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding UESecurityCapability iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	length = *(buf + decoded_size); decoded_size++;
	_5g_EASel = *(buf + decoded_size); decoded_size++; 
	_5g_IASel = *(buf + decoded_size); decoded_size++;
	Logger::nas_mm().debug("decoded UESecurityCapability EA(0x%d),IA(0x%d)", _5g_EASel,_5g_IASel);
	Logger::nas_mm().debug("decoded UESecurityCapability len(%d)", decoded_size);
	return decoded_size;
}

