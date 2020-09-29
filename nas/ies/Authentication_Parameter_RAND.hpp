#ifndef __Authentication_Parameter_RAND_H_
#define __Authentication_Parameter_RAND_H_

#include <stdint.h>

namespace nas {

	class Authentication_Parameter_RAND {
	public:
		Authentication_Parameter_RAND();
		Authentication_Parameter_RAND(uint8_t iei);
		Authentication_Parameter_RAND(const uint8_t iei, uint8_t *value);
		~Authentication_Parameter_RAND();
		//void setValue(uint8_t iei, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		uint8_t *getValue();
	private:
		uint8_t _iei;
		uint8_t _value[16];

	};



}










#endif



