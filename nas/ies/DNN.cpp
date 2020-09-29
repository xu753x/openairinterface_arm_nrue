#include "DNN.hpp"
#include "logger.hpp"
using namespace nas;

DNN::DNN(uint8_t iei) {
	_iei = iei;
}
DNN::DNN(const uint8_t iei, bstring dnn) {
	_iei = iei;
	_DNN = bstrcpy(dnn);
	length = blength(dnn) + 2;
}
DNN::DNN() {}
DNN::~DNN() {}

/*void DNN::setValue(uint8_t iei, uint8_t value) {
	_iei = iei;
	_value = value;
}*/
void DNN::getValue(bstring &dnn) {
        //dnn = _DNN;
	//dnn = bstrcpy(_DNN);
        dnn = blk2bstr((uint8_t*)bdata(_DNN)+1, blength(_DNN)-1);
}
int DNN::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding DNN iei(0x%x)", _iei);
	if (len < length) {
		Logger::nas_mm().error("len is less than %d", length);
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = (length - 2); encoded_size++;
		int size = encode_bstring(_DNN, (buf + encoded_size), len - encoded_size);
		encoded_size += size;

	}
	else {
		*(buf + encoded_size) = (length - 1); encoded_size++;
		int size = encode_bstring(_DNN, (buf + encoded_size), len - encoded_size);
		encoded_size += size;
	}
	Logger::nas_mm().debug("encoded DNN len(%d)", encoded_size);
	return encoded_size;
}

int DNN::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding DNN iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	length = *(buf + decoded_size); decoded_size++;
	decode_bstring(&_DNN, length, (buf + decoded_size), len - decoded_size);
	decoded_size += length;
	for (int i = 0; i < blength(_DNN); i++) {
	  Logger::nas_mm().debug("decoded DNN value(0x%x)", (uint8_t*)bdata(_DNN)[i]);
          //print_buffer("amf_n1", "decoded dnn bitstring", (uint8_t*)bdata(_DNN), blength(_DNN));
	}
	Logger::nas_mm().debug("decoded DNN len(%d)", decoded_size);
	return decoded_size;
}


