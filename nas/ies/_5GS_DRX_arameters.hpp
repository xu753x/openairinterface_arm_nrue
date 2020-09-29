#ifndef ___5GS_DRX_arameters_H_
#define ___5GS_DRX_arameters_H_

#include <stdint.h>

namespace nas {

	class _5GS_DRX_arameters {
	public:
		_5GS_DRX_arameters();
		_5GS_DRX_arameters(uint8_t iei);
		_5GS_DRX_arameters(const uint8_t iei, uint8_t value);
		~_5GS_DRX_arameters();
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


