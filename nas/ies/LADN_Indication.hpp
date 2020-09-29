#ifndef __LADN_Indication_H_
#define __LADN_Indication_H_
#include <iostream>
#include <vector>
#include <stdint.h>
extern "C" {
#include "bstrlib.h"
#include "TLVEncoder.h"
#include "TLVDecoder.h"
}
using namespace std;
namespace nas {

	class LADN_Indication {
	public:
		LADN_Indication();
		LADN_Indication(uint8_t iei);
		LADN_Indication(const uint8_t iei, std::vector<bstring> ladn);
		~LADN_Indication();
		void setValue(uint8_t iei, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		bool getValue(std::vector<bstring> &ladn);
	private:
		uint8_t _iei;
		uint16_t length;
		std::vector<bstring> LADN;

	};



}










#endif



