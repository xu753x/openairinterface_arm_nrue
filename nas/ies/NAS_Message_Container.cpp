#include "NAS_Message_Container.hpp"
#include "logger.hpp"
using namespace nas;

NAS_Message_Container::NAS_Message_Container(uint8_t iei) {
	_iei = iei;
}
NAS_Message_Container::NAS_Message_Container(const uint8_t iei, bstring value) {
	_iei = iei;
	_value = bstrcpy(value);
	length = blength(value) + 3;
}
NAS_Message_Container::NAS_Message_Container() {}
NAS_Message_Container::~NAS_Message_Container() {}

void NAS_Message_Container::setValue(uint8_t iei, uint8_t value) {
	_iei = iei;
	//_value = value;
}
void NAS_Message_Container::getValue(bstring &value) {
	value = bstrcpy(_value);
}


int NAS_Message_Container::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding NAS_Message_Container iei(0x%x)", _iei);
	if (len < length) {
		Logger::nas_mm().error("len is less than %d", length);
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		* (buf + encoded_size) = (length - 3) & 0x00ff; encoded_size++;
		* (buf + encoded_size) = ((length - 3) & 0xff000) >> 8; encoded_size++;
		int size = encode_bstring(_value, (buf + encoded_size), len - encoded_size);
		encoded_size += size;

    
	}
	else {
//		*(buf + encoded_size) = length - 1; encoded_size++;
//		*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded NAS_Message_Container len(%d)", encoded_size);
	return encoded_size;
}

int NAS_Message_Container::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding NAS_Message_Container iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	length = 0;
        length |= (*(buf + decoded_size))<<8; decoded_size++;
	length |= *(buf + decoded_size); decoded_size++;
	decode_bstring(&_value, length, (buf + decoded_size), len - decoded_size);
	decoded_size += length;
	for (int i = 0; i < length; i++) {
	Logger::nas_mm().debug("decoded NAS_Message_Container value(0x%x)", (uint8_t*)_value->data[i]);
	}
	Logger::nas_mm().debug("decoded NAS_Message_Container len(%d)", decoded_size);
	return decoded_size;
}



