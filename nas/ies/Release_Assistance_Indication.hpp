#ifndef __Release_Assistance_Indication_H_
#define __Release_Assistance_Indication_H_

#include <stdint.h>

namespace nas {

	class Release_Assistance_Indication {
	public:
		Release_Assistance_Indication();
		Release_Assistance_Indication(uint8_t iei);
		Release_Assistance_Indication(const uint8_t iei, uint8_t value);
		~Release_Assistance_Indication();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void setValue(uint8_t value);
		uint8_t getValue();
	private:
		uint8_t _iei;
		uint8_t _value;
	};
}














#endif




