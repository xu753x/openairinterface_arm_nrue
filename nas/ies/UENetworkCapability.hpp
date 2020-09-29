#ifndef _UENetworkCapability_H
#define _UENetworkCapability_H

#include <stdint.h>

namespace nas {

	class UENetworkCapability{
	public:
		UENetworkCapability();
		UENetworkCapability(uint8_t iei);
		~UENetworkCapability();
		UENetworkCapability(const uint8_t iei, uint8_t _5gg_EEASel, uint8_t _5gg_EIASel);
		void setEEASel(uint8_t sel);
		void setEIASel(uint8_t sel);
		uint8_t getEEASel();
		uint8_t getEIASel();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
	private:
		uint8_t _iei;
		uint8_t length;
		uint8_t _5g_EEASel;
		uint8_t _5g_EIASel;
	};







}















#endif

