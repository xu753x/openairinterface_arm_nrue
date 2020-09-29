#ifndef __UE_Status_H_
#define __UE_Status_H_

#include <stdint.h>

namespace nas {

	class UE_Status {
	public:
		UE_Status();
		UE_Status(uint8_t iei);
		UE_Status(const uint8_t iei, bool n1, bool s1);
		~UE_Status();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void setN1(bool value);
		void setS1(bool value);
		bool getN1();
		bool getS1();
	private:
		uint8_t _iei;
		bool N1;
		bool S1;
	};
}














#endif

