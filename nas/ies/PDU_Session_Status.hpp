#ifndef __PDU_Session_Status_H_
#define __PDU_Session_Status_H_

#include <stdint.h>

namespace nas {

	class PDU_Session_Status {
	public:
		PDU_Session_Status();
		PDU_Session_Status(uint8_t iei);
		PDU_Session_Status(const uint8_t iei, uint16_t value);
		~PDU_Session_Status();
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

