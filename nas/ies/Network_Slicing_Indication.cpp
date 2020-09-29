#include "Network_Slicing_Indication.hpp"
#include "logger.hpp"
using namespace nas;

Network_Slicing_Indication::Network_Slicing_Indication(uint8_t iei) {
	_iei = iei;
}
Network_Slicing_Indication::Network_Slicing_Indication(const uint8_t iei, bool dcni, bool nssci) {
	_iei = iei;
	DCNI= dcni;
	NSSCI  = nssci;
}
Network_Slicing_Indication::Network_Slicing_Indication() {}
Network_Slicing_Indication::~Network_Slicing_Indication() {}

void Network_Slicing_Indication::setDCNI(bool value) {
	DCNI = value;
}

void Network_Slicing_Indication::setNSSCI(bool value) {
	NSSCI = value;
}

bool Network_Slicing_Indication::getDCNI() {
	return DCNI;
}

bool Network_Slicing_Indication::getNSSCI() {
	return NSSCI;
}

int Network_Slicing_Indication::encode2buffer(uint8_t *buf, int len) {
	Logger::nas_mm().debug("encoding Network_Slicing_Indication iei(0x%x)", _iei);
	if (len < 1) {
	//	Logger::nas_mm().error("len is less than %d", length);
		return 0;
	}
	uint8_t octet = 0;
	if (!(_iei & 0x0f)) {
		//octet = (0x0f) & ((tsc << 3) | key_id);
		//*buf = octet;
		//Logger::nas_mm().debug("encoded Payload_Container_Type IE(len(1/2 octet))");
		//return 0;
	}
	else {
		octet = (_iei << 4) | (DCNI<<1)|NSSCI;
	Logger::nas_mm().debug("decoded Network_Slicing_Indication DCNI(0x%x) NSSCI(0x%x)", octet,NSSCI);
		*buf = octet;
		Logger::nas_mm().debug("encoded Network_Slicing_Indication IE(len(1 octet))");
		return 1;
	}
}

int Network_Slicing_Indication::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
	if (len < 1) {
	Logger::nas_mm().error("len is less than one");
	return 0;
	}
	else {
	uint8_t octet = (*buf);
	if (is_option) {
		_iei = (octet & 0xf0) >> 4;
	}
	else {
		_iei = 0;
	}
	DCNI = 0; NSSCI = 0;
	DCNI = octet&0x02; 
	NSSCI =octet& 0x01; 
	Logger::nas_mm().debug("decoded Network_Slicing_Indication DCNI(0x%x) NSSCI(0x%x)", DCNI,NSSCI);
	return 1;
}
}


