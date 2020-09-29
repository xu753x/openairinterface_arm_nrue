#ifndef __PLMN_List_H_
#define __PLMN_List_H_

#include <stdint.h>

namespace nas {

	class PLMN_List {
	public:
		PLMN_List();
		PLMN_List(uint8_t iei);
		PLMN_List(const uint8_t iei, uint8_t MNC_MCC1, uint8_t MNC_MCC2, uint8_t MNC_MCC3);
		~PLMN_List();
		void setMNC_MCC1(uint8_t iei, uint8_t value);
		void setMNC_MCC2(uint8_t iei, uint8_t value);
		void setMNC_MCC3(uint8_t iei, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		uint8_t getMNC_MCC1();
		uint8_t getMNC_MCC2();
		uint8_t getMNC_MCC3();
	private:
		uint8_t _iei;
		uint8_t _MNC_MCC1;
		uint8_t _MNC_MCC2;
		uint8_t _MNC_MCC3;
	};
}














#endif


