#include "Authentication_Parameter_RAND.hpp"
#include "logger.hpp"
using namespace nas;

Authentication_Parameter_RAND::Authentication_Parameter_RAND(uint8_t iei) {
	_iei = iei;
}
Authentication_Parameter_RAND::Authentication_Parameter_RAND(const uint8_t iei, uint8_t *value) {
	_iei = iei;
	for(int i=0;i<16;i++){this->_value[i] = value[i];}
}
Authentication_Parameter_RAND::Authentication_Parameter_RAND() {}
Authentication_Parameter_RAND::~Authentication_Parameter_RAND() {}

uint8_t *Authentication_Parameter_RAND::getValue() {
	return _value;
}
int Authentication_Parameter_RAND::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding Authentication_Parameter_RAND iei(0x%x)", _iei);
	if (len < 17) {
		Logger::nas_mm().error("len is less than 17");
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
                memcpy((void*)(buf+encoded_size), (void*)_value, 16); encoded_size += 16;
		//int i = 0;
		//for (int i = 0; i < 16; i++) {
		//	*(buf + encoded_size) = _value[i]; encoded_size++;
		//}
		return encoded_size;
	}
	else {
//		*(buf + encoded_size) = length - 1; encoded_size++;
//		*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded Authentication_Parameter_RAND len(%d)", encoded_size);
	return encoded_size;
}

int Authentication_Parameter_RAND::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding Authentication_Parameter_RAND iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	for(int i=0;i<16;i++){
		_value[i] = *(buf + decoded_size); decoded_size++;
	}
	for (int j = 0; j < 16; j++) {
		Logger::nas_mm().debug("decoded Authentication_Parameter_RAND value(0x%2x)", _value[j]);
	}
	Logger::nas_mm().debug("decoded Authentication_Parameter_RAND len(%d)", decoded_size);
	return decoded_size;
}



