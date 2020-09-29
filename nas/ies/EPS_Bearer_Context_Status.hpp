#ifndef __EPS_Bearer_Context_Status_H_
#define __EPS_Bearer_Context_Status_H_

#include <stdint.h>

namespace nas {

	class EPS_Bearer_Context_Status {
	public:
		EPS_Bearer_Context_Status();
		EPS_Bearer_Context_Status(uint8_t iei);
		EPS_Bearer_Context_Status(const uint8_t iei, uint16_t value);
		~EPS_Bearer_Context_Status();
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


