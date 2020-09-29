#include "LADN_Indication.hpp"
#include "logger.hpp"
using namespace nas;

LADN_Indication::LADN_Indication(uint8_t iei) {
	_iei = iei;
}
LADN_Indication::LADN_Indication(const uint8_t iei, std::vector<bstring> ladn) {
	_iei = iei;
	length = 3;
	LADN.assign(ladn.begin(), ladn.end());
   for (int i = 0; i < ladn.size(); i++) {
	 length = length + 1+blength(ladn.at(i));
	}
}
LADN_Indication::LADN_Indication() {}
LADN_Indication::~LADN_Indication() {}

void LADN_Indication::setValue(uint8_t iei, uint8_t value) {
	_iei = iei;
	//_value = value;
}
bool LADN_Indication::getValue(std::vector<bstring> &ladn){
	ladn.assign(LADN.begin(), LADN.end());
	return 0;
}

int LADN_Indication::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding LADN_Indication iei(0x%x)", _iei);
	if (len < length) {
		Logger::nas_mm().error("len is less than %d", length);
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = (length - 3)&0x00ff; encoded_size++;
		*(buf + encoded_size) = ((length - 3) & 0xff00)>>8; encoded_size++;
		for (int i = 0; i < LADN.size(); i++) {
			*(buf + encoded_size) = blength(LADN.at(i));encoded_size++;
			encoded_size+=encode_bstring(LADN.at(i), (buf + encoded_size), len - encoded_size);
		}
	}
	else {
//		*(buf + encoded_size) = length - 1; encoded_size++;
//		*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded LADN_Indication len(%d)", encoded_size);
	return encoded_size;
}

int LADN_Indication::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding LADN_Indication iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	length = 0;
	length |= *(buf + decoded_size); decoded_size++;
	length |= (*(buf + decoded_size))<<8; decoded_size++;
	Logger::nas_mm().debug("decoded LADN_Indication len(%d)", length);
	int LEAGTH = length;
	uint8_t len_dnn;
	bstring dnn;
	while (LEAGTH) {
		len_dnn= *(buf + decoded_size); decoded_size++; LEAGTH--;
		decode_bstring(&dnn, len_dnn, (buf + decoded_size), len - decoded_size);
		decoded_size += len_dnn;
		LEAGTH-= len_dnn;
		LADN.insert(LADN.end(), dnn);
	}
	for (int i = 0; i < LADN.size(); i++) {
		for (int j = 0; j < blength(LADN.at(i)); j++) {
			Logger::nas_mm().debug("decoded LADN_Indication value(0x%x)", (uint8_t*)LADN.at(i)->data[j]);
		}
	}
	Logger::nas_mm().debug("decoded LADN_Indication len(%d)", decoded_size);
	return decoded_size;
}



