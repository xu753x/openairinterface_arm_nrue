#ifndef __Request_Type_H_
#define __Request_Type_H_

#include <stdint.h>

namespace nas {

	class Request_Type {
	public:
		Request_Type();
		Request_Type(uint8_t iei);
		Request_Type(const uint8_t iei, uint8_t value);
		~Request_Type();
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




