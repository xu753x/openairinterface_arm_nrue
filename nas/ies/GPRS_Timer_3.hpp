#ifndef __GPRS_Timer_3_H_
#define __GPRS_Timer_3_H_

#include <stdint.h>

namespace nas {

	class GPRS_Timer_3 {
	public:
		GPRS_Timer_3();
		GPRS_Timer_3(uint8_t iei);
		GPRS_Timer_3(const uint8_t iei,uint8_t unit, uint8_t value);
		~GPRS_Timer_3();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void setValue(uint8_t unit,uint8_t value);
		uint8_t getValue();
		uint8_t getUnit();
	private:
		uint8_t _iei;
		uint8_t UNIT;
		uint8_t _value;
	};
}














#endif



