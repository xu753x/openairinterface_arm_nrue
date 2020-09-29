#ifndef __5GMM_Cause_H
#define __5GMM_Cause_H

#include <stdint.h>

namespace nas {

	class _5GMM_Cause {
	public:
		_5GMM_Cause();
		_5GMM_Cause(uint8_t _iei,uint8_t value);
		~_5GMM_Cause();
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



