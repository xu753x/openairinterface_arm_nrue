#ifndef __EPS_NAS_Security_Algorithms_H
#define __EPS_NAS_Security_Algorithms_H

#include <stdint.h>

namespace nas {

	class EPS_NAS_Security_Algorithms {
	public:
		EPS_NAS_Security_Algorithms();
		~EPS_NAS_Security_Algorithms();
		EPS_NAS_Security_Algorithms(uint8_t iei, uint8_t ciphering, uint8_t  integrity_protection);
		void setCIPHERING(uint8_t value);
		void setINTEGRITY_PROTECTION(uint8_t value);
		uint8_t getCIPHERING();
		uint8_t getINTEGRITY_PROTECTION();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
	private:
		uint8_t _iei;
		uint8_t CIPHERING; 
		uint8_t INTEGRITY_PROTECTION;
	};







}















#endif



