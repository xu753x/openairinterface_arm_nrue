#include "PLMN_List.hpp"
#include "logger.hpp"
using namespace nas;

PLMN_List::PLMN_List(uint8_t iei) {
	_iei = iei;
}
PLMN_List::PLMN_List(const uint8_t iei, uint8_t MNC_MCC1, uint8_t MNC_MCC2, uint8_t MNC_MCC3) {
	_iei = iei;
	_MNC_MCC1 = MNC_MCC1;
	_MNC_MCC2 = MNC_MCC2;
	_MNC_MCC3 = MNC_MCC3;
}
PLMN_List::PLMN_List() {}
PLMN_List::~PLMN_List() {}

void PLMN_List::setMNC_MCC1(uint8_t iei, uint8_t value) {
	_iei = iei;
	_MNC_MCC1 = value;
}
void PLMN_List::setMNC_MCC2(uint8_t iei, uint8_t value) {
	_iei = iei;
	_MNC_MCC2 = value;
}
void PLMN_List::setMNC_MCC3(uint8_t iei, uint8_t value) {
	_iei = iei;
	_MNC_MCC3 = value;
}

uint8_t PLMN_List::getMNC_MCC1() {
	return _MNC_MCC1;
}
uint8_t PLMN_List::getMNC_MCC2() {
	return _MNC_MCC2;
}
uint8_t PLMN_List::getMNC_MCC3() {
	return _MNC_MCC3;
}
int PLMN_List::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding PLMN_List iei(0x%x)", _iei);
	if (len < 5) {
		Logger::nas_mm().error("len is less than 5");
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = 3; encoded_size++;
		*(buf + encoded_size) = (_MNC_MCC1 & 0x0F)|((_MNC_MCC2 & 0x0F)<<4);     encoded_size++;
		*(buf + encoded_size) = _MNC_MCC3;                                     encoded_size++;
		*(buf + encoded_size) = ((_MNC_MCC1 & 0xF0)>>4) | (_MNC_MCC2 & 0xF0); encoded_size++;
	}
	else {
	//	*(buf + encoded_size) = length - 1; encoded_size++;
	//	*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded PLMN_List len(%d)", encoded_size);
	return encoded_size;
}

int PLMN_List::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding PLMN_List iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	decoded_size++;
	_MNC_MCC1 = 0x00;
	_MNC_MCC2 = 0x00;
	_MNC_MCC3 = 0x00;
	_MNC_MCC1 |= *(buf + decoded_size) & 0x0F;
	_MNC_MCC2 |=( *(buf + decoded_size) & 0xF0)>>4; decoded_size++;
	_MNC_MCC3 = *(buf + decoded_size); decoded_size++; 
	_MNC_MCC1 |= (*(buf + decoded_size) & 0x0F)<<4;
	_MNC_MCC2 |= *(buf + decoded_size) & 0xF0; decoded_size++;
	Logger::nas_mm().debug("decoded PLMN_List MNC_MCC1(0x%x),MNC_MCC2(0x%x),MNC_MCC3(0x%x)", _MNC_MCC1,_MNC_MCC2,_MNC_MCC3);
	Logger::nas_mm().debug("decoded PLMN_List len(%d)", decoded_size);
	return decoded_size;
}


