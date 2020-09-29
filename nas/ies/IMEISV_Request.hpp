#ifndef __IMEISV_Request_H_
#define __IMEISV_Request_H_

#include <stdint.h>

namespace nas {

	class IMEISV_Request {
	public:
		IMEISV_Request();
		IMEISV_Request(uint8_t iei);
		IMEISV_Request(const uint8_t iei, uint8_t value);
		~IMEISV_Request();
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



