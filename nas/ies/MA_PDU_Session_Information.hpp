#ifndef __MA_PDU_Session_Information_H_
#define __MA_PDU_Session_Information_H_

#include <stdint.h>

namespace nas {

	class MA_PDU_Session_Information {
	public:
		MA_PDU_Session_Information();
		MA_PDU_Session_Information(uint8_t iei);
		MA_PDU_Session_Information(const uint8_t iei, uint8_t value);
		~MA_PDU_Session_Information();
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




