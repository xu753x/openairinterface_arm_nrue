#ifndef __UE_Radio_Capability_ID_H_
#define __UE_Radio_Capability_ID_H_

#include <stdint.h>

namespace nas {

	class UE_Radio_Capability_ID {
	public:
		UE_Radio_Capability_ID();
		UE_Radio_Capability_ID(uint8_t iei);
		UE_Radio_Capability_ID(const uint8_t iei, uint8_t value);
		~UE_Radio_Capability_ID();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void setValue(uint8_t value);
		uint8_t getValue();
	private:
		uint8_t _iei;
		uint8_t _value;
	};
}














#endif



