#ifndef __S_NSSAI_H_
#define __S_NSSAI_H_

#include <stdint.h>
#include "nas_ie_header.hpp"
namespace nas {

        class S_NSSAI {
        public:
			S_NSSAI();
			S_NSSAI(uint8_t iei);
			S_NSSAI(const uint8_t iei, SNSSAI_s snssai);
                ~S_NSSAI();
                void setS_NSSAI(SNSSAI_s snssai);
                int encode2buffer(uint8_t *buf, int len);
                int decodefrombuffer(uint8_t *buf, int len, bool is_option);
                void getValue(SNSSAI_s &snssai);
        private:
                uint8_t _iei;
                uint8_t length;
				SNSSAI_s SNSSAI;

        };

}












#endif



