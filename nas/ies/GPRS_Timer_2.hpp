#ifndef __GPRS_Timer_2_H_
#define __GPRS_Timer_2_H_

#include <stdint.h>

namespace nas {

	class GPRS_Timer_2 {
	public:
		GPRS_Timer_2();
		GPRS_Timer_2(uint8_t iei);
		GPRS_Timer_2(const uint8_t iei, uint8_t value);
		~GPRS_Timer_2();
		void setIEI(uint8_t iei);
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


