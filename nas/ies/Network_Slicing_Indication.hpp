#ifndef __Network_Slicing_Indication_H_
#define __Network_Slicing_Indication_H_

#include <stdint.h>

namespace nas {

	class Network_Slicing_Indication {
	public:
		Network_Slicing_Indication();
		Network_Slicing_Indication(uint8_t iei);
		Network_Slicing_Indication(const uint8_t iei, bool dcni, bool nssci);
		~Network_Slicing_Indication();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void setDCNI(bool value);
		void setNSSCI(bool value);
		bool getDCNI();
		bool getNSSCI();
	private:
		uint8_t _iei;
		bool NSSCI;
		bool DCNI;
	};
}














#endif


