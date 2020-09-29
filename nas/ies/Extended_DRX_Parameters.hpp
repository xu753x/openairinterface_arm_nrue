#ifndef __Extended_DRX_Parameters_H_
#define __Extended_DRX_Parameters_H_

#include <stdint.h>

namespace nas {

	class Extended_DRX_Parameters {
	public:
		Extended_DRX_Parameters();
		Extended_DRX_Parameters(uint8_t iei);
		Extended_DRX_Parameters(const uint8_t iei, uint8_t paging_time, uint8_t value);
		~Extended_DRX_Parameters();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void setValue(uint8_t value);
		uint8_t  getValue();
		void setPaging_time(uint8_t value);
		uint8_t getPaging_time();
	private:
		uint8_t _iei;
		uint8_t _paging_time;
		uint8_t _value;
	};
}














#endif



