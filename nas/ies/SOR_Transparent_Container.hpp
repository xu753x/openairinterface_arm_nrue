#ifndef __SOR_Transparent_Container_H_
#define __SOR_Transparent_Container_H_

#include <stdint.h>

namespace nas {

	class SOR_Transparent_Container {
	public:
		SOR_Transparent_Container();
		SOR_Transparent_Container(uint8_t iei);
		SOR_Transparent_Container(const uint8_t iei, uint8_t header, uint8_t *value);
		~SOR_Transparent_Container();
		//void setValue(uint8_t iei, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		uint8_t getValue();
	private:
		uint8_t _iei;
		uint8_t HEADER;
		uint8_t _value[16];

	};



}










#endif





