#ifndef ___PDU_Session_Identity_2_H_
#define ___PDU_Session_Identity_2_H_

#include <stdint.h>

namespace nas {

	class PDU_Session_Identity_2 {
	public:
		PDU_Session_Identity_2();
		PDU_Session_Identity_2(uint8_t iei);
		PDU_Session_Identity_2(const uint8_t iei, uint8_t value);
		~PDU_Session_Identity_2();
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



