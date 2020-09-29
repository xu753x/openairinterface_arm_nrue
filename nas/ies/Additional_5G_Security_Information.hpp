#ifndef __Additional_5G_Security_Information_H_
#define __Additional_5G_Security_Information_H_

#include <stdint.h>

namespace nas {

	class Additional_5G_Security_Information {
	public:
		Additional_5G_Security_Information();
		Additional_5G_Security_Information(uint8_t iei);
		Additional_5G_Security_Information(const uint8_t iei, bool RINMR, bool HDP);
		~Additional_5G_Security_Information();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void setRINMR(bool value);
		void setHDP(bool value);
		bool getRINMR();
		bool getHDP();
	private:
		uint8_t _iei;
		bool RINMR;
		bool HDP;
	};
}














#endif



