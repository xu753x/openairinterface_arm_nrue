#ifndef __Payload_Container_H_
#define __Payload_Container_H_
#include <iostream>
#include <stdint.h>
#include <vector>
#include "nas_ie_header.hpp"
extern "C" {
#include "bstrlib.h"
#include "TLVEncoder.h"
#include "TLVDecoder.h"
}
namespace nas {
	class Payload_Container {
	public:
		Payload_Container();
		Payload_Container(uint8_t iei);
		Payload_Container(uint8_t iei, bstring b);
		Payload_Container(const uint8_t iei, std::vector<PayloadContainerEntry> content);
		~Payload_Container();
		void setValue(uint8_t iei, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option, uint8_t type);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void getValue(std::vector<PayloadContainerEntry> &content);
                void getValue(bstring &cnt);
	private:
		uint8_t _iei;
		uint16_t length;
                bstring content;
		std::vector<PayloadContainerEntry> CONTENT;

	};



}










#endif



