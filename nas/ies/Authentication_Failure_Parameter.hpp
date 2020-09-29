#ifndef __Authentication_Failure_Parameter_H_
#define __Authentication_Failure_Parameter_H_
#include <iostream>
#include <stdint.h>
extern "C" {
#include "bstrlib.h"
#include "TLVEncoder.h"
#include "TLVDecoder.h"
}
namespace nas {

	class Authentication_Failure_Parameter {
	public:
		Authentication_Failure_Parameter();
		Authentication_Failure_Parameter(uint8_t iei);
		Authentication_Failure_Parameter(const uint8_t iei,  bstring auts);
		~Authentication_Failure_Parameter();
		//void setValue(uint8_t iei, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void getValue(bstring &auts);
	private:
		uint8_t _iei;
		uint8_t length;
		bstring value;

	};



}










#endif







