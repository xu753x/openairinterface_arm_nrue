#include "UE_Status.hpp"
#include "logger.hpp"
using namespace nas;

UE_Status::UE_Status(uint8_t iei) {
	_iei = iei;
}
UE_Status::UE_Status(const uint8_t iei, bool n1, bool s1) {
	_iei = iei;
	S1 = s1;
	N1 = n1;
}
UE_Status::UE_Status() {}
UE_Status::~UE_Status() {}

void UE_Status::setS1(bool value) {
	S1 = value;
}

void UE_Status::setN1(bool value) {
	N1 = value;
}

bool UE_Status::getS1() {
	return S1;
}

bool UE_Status::getN1() {
	return N1;
}

int UE_Status::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding UE_Status iei(0x%x)", _iei);
	if (len < 3) {
	//	Logger::nas_mm().error("len is less than %d", length);
		return 0;
	}
	int encoded_size = 0;
	if (_iei) {
		*(buf + encoded_size) = _iei; encoded_size++;
		*(buf + encoded_size) = 3 - 2;     
 		   encoded_size++;
		*(buf + encoded_size) = 0x00| S1 |N1<<1;           encoded_size++;
	}
	else {
	//	*(buf + encoded_size) = length - 1; encoded_size++;
	//	*(buf + encoded_size) = _value; encoded_size++; encoded_size++;
	}
	Logger::nas_mm().debug("encoded UE_Status len(%d)", encoded_size);
	return encoded_size;
}

int UE_Status::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	Logger::nas_mm().debug("decoding UE_Status iei(0x%x)", *buf);
	int decoded_size = 0;
	if (is_option) {
		decoded_size++;
	}
	S1 = 0; N1 = 0;
	//length = *(buf + decoded_size);
	 decoded_size++;
	N1 = *(buf + decoded_size)&0x02; 
	S1 = *(buf + decoded_size) & 0x01; decoded_size++;
	Logger::nas_mm().debug("decoded UE_Status N1(0x%x) S1(0x%x)", N1,S1);
	Logger::nas_mm().debug("decoded UE_Status len(%d)", decoded_size);
	return decoded_size;
}

