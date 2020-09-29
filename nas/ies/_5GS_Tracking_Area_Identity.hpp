#ifndef __5GS_Tracking_Area_Identity_H_
#define __5GS_Tracking_Area_Identity_H_

#include <stdint.h>

namespace nas {

	class _5GS_Tracking_Area_Identity {
	public:
		_5GS_Tracking_Area_Identity();
		_5GS_Tracking_Area_Identity(uint8_t iei);
		_5GS_Tracking_Area_Identity(const uint8_t iei, uint8_t MNC_MCC1, uint8_t MNC_MCC2, uint8_t MNC_MCC3, uint32_t TAC);
		~_5GS_Tracking_Area_Identity();
		void setMNC_MCC1(uint8_t iei, uint8_t value);
		void setMNC_MCC2(uint8_t iei, uint8_t value);
		void setMNC_MCC3(uint8_t iei, uint8_t value);
		void setTAC(uint8_t iei, uint32_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		uint8_t getMNC_MCC1();
		uint8_t getMNC_MCC2();
		uint8_t getMNC_MCC3();
		uint32_t getTAC();
	private:
		uint8_t _iei;
		uint8_t _MNC_MCC1;
		uint8_t _MNC_MCC2;
		uint8_t _MNC_MCC3;
		uint8_t _TAC1;
		uint8_t _TAC2;
		uint8_t _TAC3;
		
	};
}














#endif

