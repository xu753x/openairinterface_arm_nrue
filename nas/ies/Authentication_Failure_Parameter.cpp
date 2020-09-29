#include "Authentication_Failure_Parameter.hpp"
#include "logger.hpp"
using namespace nas;

Authentication_Failure_Parameter::Authentication_Failure_Parameter(uint8_t iei) {
	_iei = iei;
}
Authentication_Failure_Parameter::Authentication_Failure_Parameter(const uint8_t iei, bstring auts) {
	_iei = iei;
	value = bstrcpy(auts);
	length = blength(auts) + 2;
}
Authentication_Failure_Parameter::Authentication_Failure_Parameter() {}
Authentication_Failure_Parameter::~Authentication_Failure_Parameter() {}

void Authentication_Failure_Parameter::getValue(bstring &auts) {
	auts = bstrcpy(value);
}
int Authentication_Failure_Parameter::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding Authentication_Failure_Parameter iei(0x%x)", _iei);
	if (len < length) {
		Logger::nas_mm().error("len is less than %x",length);
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = length-2; encoded_size++;
		int size = encode_bstring(value, (buf + encoded_size), len - encoded_size);
		encoded_size += size;
		return encoded_size;
	}
	else {
//		*(buf + encoded_size) = length - 1; encoded_size++;
//		*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded Authentication_Failure_Parameter len(%d)", encoded_size);
	return encoded_size;
}

int Authentication_Failure_Parameter::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding Authentication_Failure_Parameter iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	length = *(buf + decoded_size); decoded_size++;
        Logger::nas_mm().debug("decoded IE Authentication_Failure_Parameter length(%d)", length);
	decode_bstring(&value, length, (buf + decoded_size), len - decoded_size);
	decoded_size += length;
	for (int i = 0; i < length; i++) {
		Logger::nas_mm().debug("decoded Authentication_Failure_Parameter value(0x%x)", (uint8_t*)value->data[i]);
	}
	Logger::nas_mm().debug("decoded Authentication_Failure_Parameter len(%d)", decoded_size);
	return decoded_size;
}







