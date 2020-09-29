#ifndef __DNN_H_
#define __DNN_H_
#include <iostream>
#include <stdint.h>
extern "C" {
#include "bstrlib.h"
#include "TLVEncoder.h"
#include "TLVDecoder.h"
}
namespace nas {

	class DNN {
	public:
		DNN();
		DNN(uint8_t iei);
		DNN(const uint8_t iei, bstring dnn);
		~DNN();
		//void setValue(uint8_t iei, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void getValue(bstring &dnn);
	private:
		uint8_t _iei;
		uint8_t length;
		bstring _DNN;

	};



}










#endif


