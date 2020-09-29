#ifndef ___5GS_Identity_Type_H_
#define ___5GS_Identity_Type_H_

#include <stdint.h>

namespace nas {

	class _5GS_Identity_Type {
	public:
		_5GS_Identity_Type();
		_5GS_Identity_Type(uint8_t iei);
		_5GS_Identity_Type(const uint8_t iei, uint8_t value);
		~_5GS_Identity_Type();
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



