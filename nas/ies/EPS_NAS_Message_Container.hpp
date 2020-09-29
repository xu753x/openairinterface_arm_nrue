#ifndef __EPS_NAS_Message_Container_H_
#define __EPS_NAS_Message_Container_H_
#include <iostream>
#include <stdint.h>
extern "C" {
#include "bstrlib.h"
#include "TLVEncoder.h"
#include "TLVDecoder.h"
}
namespace nas {

	class EPS_NAS_Message_Container {
	public:
		EPS_NAS_Message_Container();
		EPS_NAS_Message_Container(uint8_t iei);
		EPS_NAS_Message_Container(const uint8_t iei, bstring value);
		~EPS_NAS_Message_Container();
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



