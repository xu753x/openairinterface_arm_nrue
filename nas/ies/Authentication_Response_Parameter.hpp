#ifndef __Authentication_Response_Parameter_H_
#define __Authentication_Response_Parameter_H_
#include <iostream>
 #include <stdint.h>
  extern "C"{
  #include "bstrlib.h"
 #include "TLVEncoder.h"
 #include "TLVDecoder.h"
  }

namespace nas {

	class Authentication_Response_Parameter {
	public:
		Authentication_Response_Parameter();
		Authentication_Response_Parameter(uint8_t iei);
		Authentication_Response_Parameter(const uint8_t iei, bstring para);
		~Authentication_Response_Parameter();
		//void setValue(uint8_t iei, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void getValue(bstring &para);
	private:
		uint8_t _iei;
		bstring PARA;

	};



}










#endif





