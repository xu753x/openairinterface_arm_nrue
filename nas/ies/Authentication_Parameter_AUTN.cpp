#include "Authentication_Parameter_AUTN.hpp"
#include "logger.hpp"
using namespace nas;

Authentication_Parameter_AUTN::Authentication_Parameter_AUTN(uint8_t iei) {
	_iei = iei;
}
Authentication_Parameter_AUTN::Authentication_Parameter_AUTN(const uint8_t iei,uint8_t *value) {
	_iei = iei;
	for(int i=0;i<16;i++){
	this->_value[i] =value[i];}
}
Authentication_Parameter_AUTN::Authentication_Parameter_AUTN() {}
Authentication_Parameter_AUTN::~Authentication_Parameter_AUTN() {}

uint8_t *Authentication_Parameter_AUTN::getValue() {
	//for (int j = 0; j < 16; j++) {
	//	Logger::nas_mm().debug("decoded Authentication_Response_Parameter value(0x%2x)", _value[j]);
	//}
	return _value;}

int Authentication_Parameter_AUTN::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding Authentication_Parameter_AUTN iei(0x%x)", _iei);
	if (len < 18) {
		Logger::nas_mm().error("len is less than 18");
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = 16; encoded_size++;
		for (int i = 0; i < 16;i++) {
			*(buf + encoded_size) = _value[i]; encoded_size++;
		}
		return encoded_size;
	}
	else {
//		*(buf + encoded_size) = length - 1; encoded_size++;
//		*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded Authentication_Parameter_AUTN len(%d)", encoded_size);
	return encoded_size;
}

int Authentication_Parameter_AUTN::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding Authentication_Parameter_AUTN iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	decoded_size++;
	for(int  i=0;i<16;i++) {
		_value[i] = *(buf + decoded_size); decoded_size++;
	}
	for(int j=0;j<16;j++){
	Logger::nas_mm().debug("decoded Authentication_Parameter_AUTN value(0x%2x)", _value[j]);}
	Logger::nas_mm().debug("decoded Authentication_Parameter_AUTN len(%d)", decoded_size);
	return decoded_size;
}




