#ifndef _NSSAI_Inclusion_Mode_H
#define _NSSAI_Inclusion_Mode_H

#include <stdint.h>

namespace nas {

	class NSSAI_Inclusion_Mode {
	public:
		NSSAI_Inclusion_Mode();
		NSSAI_Inclusion_Mode(const uint8_t iei, uint8_t value);
		~NSSAI_Inclusion_Mode();
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



