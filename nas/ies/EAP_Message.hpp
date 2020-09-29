#ifndef __EAP_Message_H_
#define __EAP_Message_H_

#include <iostream>
 #include <stdint.h>
  extern "C"{
  #include "bstrlib.h"
 #include "TLVEncoder.h"
 #include "TLVDecoder.h"
  }
namespace nas {

	class EAP_Message {
	public:
		EAP_Message();
		EAP_Message(uint8_t iei);
		EAP_Message(const uint8_t iei, bstring eap);
		~EAP_Message();
		//void setValue(uint8_t iei, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void getValue(bstring &eap);
	private:
		uint8_t _iei;
		uint16_t length;
		bstring EAP;

	};



}










#endif



