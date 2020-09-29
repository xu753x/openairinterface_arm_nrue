#ifndef __UplinkDataStatus_H_
#define __UplinkDataStatus_H_

#include <stdint.h>

namespace nas {

	class UplinkDataStatus {
	public:
		UplinkDataStatus();
		UplinkDataStatus(uint8_t);
		UplinkDataStatus(const uint8_t iei, uint16_t value);
		~UplinkDataStatus();
		void setValue(uint8_t iei, uint16_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		uint16_t getValue();
	private:
		uint8_t _iei;
		uint8_t length;
		uint16_t _value;
		
	};


}











#endif

