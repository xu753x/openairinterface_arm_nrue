#ifndef __Rejected_NSSAI_H_
#define __Rejected_NSSAI_H_

#include <stdint.h>

namespace nas {

        class Rejected_NSSAI {
        public:
			Rejected_NSSAI();
			Rejected_NSSAI(uint8_t iei);
			Rejected_NSSAI(const uint8_t iei, uint8_t cause, uint8_t SST);
            ~Rejected_NSSAI();
            void setSST(uint8_t SST);
			void setCause(uint8_t SST);
            int encode2buffer(uint8_t *buf, int len);
            int decodefrombuffer(uint8_t *buf, int len, bool is_option);
            uint8_t getSST();
			uint8_t getCause();
        private:
                uint8_t _iei;
                uint8_t length;
                uint8_t _s_nssai_length;
				uint8_t _cause;
                uint8_t _s_nssai_SST;

        };

}












#endif


