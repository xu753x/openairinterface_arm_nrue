#ifndef _MICO_Indication_H
#define _MICO_Indication_H

#include <stdint.h>

namespace nas {

	class MICO_Indication {
	public:
		MICO_Indication();
		MICO_Indication(const uint8_t _iei, bool sprti, bool raai);
		MICO_Indication(bool sprti, bool raai);
		~MICO_Indication();

		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);

		void setSPRTI(bool value);
		void setRAAI(bool value);
		bool getRAAI();
		bool getSPRTI();
	private:
		uint8_t iei;
		bool SPRTI;
		bool RAAI;
	};





}






#endif

