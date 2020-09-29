#ifndef __NAS_Message_Container_H_
#define __NAS_Message_Container_H_
#include <iostream>
 #include <stdint.h>
  extern "C"{
  #include "bstrlib.h"
 #include "TLVEncoder.h"
 #include "TLVDecoder.h"
  }


namespace nas {

	class NAS_Message_Container {
	public:
		NAS_Message_Container();
		NAS_Message_Container(uint8_t iei);
		NAS_Message_Container(const uint8_t iei, bstring value);
		~NAS_Message_Container();
		void setValue(uint8_t iei, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void getValue(bstring &value);
	private:
		uint8_t _iei;
		uint16_t length;
		bstring _value;

	};



}










#endif



