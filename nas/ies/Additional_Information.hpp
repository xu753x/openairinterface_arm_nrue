#ifndef __Additional_Information_H_
#define __Additional_Information_H_

#include <stdint.h>

namespace nas {

	class Additional_Information {
	public:
		Additional_Information();
		Additional_Information(uint8_t iei);
		Additional_Information(const uint8_t iei, uint8_t _length, uint8_t value);
		~Additional_Information();
		void setValue(uint8_t iei, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		uint8_t getValue();
	private:
		uint8_t _iei;
		uint8_t length;
		uint8_t _value;

	};



}










#endif

