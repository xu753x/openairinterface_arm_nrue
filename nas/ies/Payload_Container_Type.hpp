#ifndef _Payload_Container_Type_H
#define _Payload_Container_Type_H

#include <stdint.h>

namespace nas {

	class Payload_Container_Type {
	public:
		Payload_Container_Type();
		Payload_Container_Type(const uint8_t iei, uint8_t value);
		~Payload_Container_Type();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void setValue(const uint8_t value);
		uint8_t getValue();
	private:
		uint8_t _iei;	
		uint8_t _value;
	};





}






#endif


