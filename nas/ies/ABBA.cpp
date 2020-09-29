#include "ABBA.hpp"
#include "logger.hpp"
using namespace nas;

ABBA::ABBA(uint8_t iei) {
	_iei = iei;
}
ABBA::ABBA(const uint8_t iei, uint8_t length, uint8_t *value) {
	_iei = iei;
	for(int i=0;i<length;i++){
	this->_value[i] =value[i];}
	_length = length;
}
ABBA::ABBA() {}
ABBA::~ABBA() {}

uint8_t ABBA::getValue() {
	for (int j = 0; j < _length; j++) {
		Logger::nas_mm().debug("decoded ABBA value(0x%2x)", _value[j]);
	}
	return 1;
}
int ABBA::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding ABBA iei(0x%x)", _iei);
	if (len < _length) {
		Logger::nas_mm().error("len is less than %d", _length);
		return 0;
	}
	int encoded_size = 0;
	if (_iei) { //option
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = _length - 2; encoded_size++;
		int i = 0;
		while ((_length - 2) != 0) {
			*(buf + encoded_size) = _value[i]; encoded_size++;
			_length--; i++;
		}
	}
	else {
                Logger::nas_mm().debug("length(%d)", _length);
		*(buf + encoded_size) = _length; encoded_size++;
		int i = 0;
		while (_length != 0) {
			*(buf + encoded_size) = _value[i]; encoded_size++;
			_length--; i++;
		}
	}
	Logger::nas_mm().debug("encoded ABBA len(%d)", encoded_size);
	return encoded_size;
}

int ABBA::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding ABBA iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}       
	_length=0x00;
	_value[255]={} ;
	_length = *(buf + decoded_size); decoded_size++;
	int i = 0;uint8_t Length=_length;
	while (Length != 0) {
		_value[i] = *(buf + decoded_size); decoded_size++;
		Length--; i++;
	}
	for(int j=0;j<_length;j++){
	Logger::nas_mm().debug("decoded ABBA value(0x%4x),length(0x%4x)", _value[j],_length);}
	Logger::nas_mm().debug("decoded ABBA len(%d)", decoded_size);
	return decoded_size;
}





