#ifndef ___5GS_Update_Type_H
#define ___5GS_Update_Type_H

#include <stdint.h>

namespace nas {

	class _5GS_Update_Type {
	public:
		_5GS_Update_Type();
		_5GS_Update_Type(uint8_t iei);
		~_5GS_Update_Type();
		_5GS_Update_Type(const uint8_t iei, uint8_t eps_PNB_CIoT, uint8_t _5gs_PNB_CIoT,bool ng_RAN, bool sms);
		void setEPS_PNB_CIoT(uint8_t value);
		void set_5GS_PNB_CIoT(uint8_t value);
		void setNG_RAN(uint8_t value);
		void setSMS(uint8_t value);
		uint8_t getEPS_PNB_CIoT();
		uint8_t get_5GS_PNB_CIoT();
		bool getNG_RAN();
		bool getSMS();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
	private:
		uint8_t _iei;
		uint8_t length;
		uint8_t EPS_PNB_CIoT;
		uint8_t _5GS_PNB_CIoT;
		bool NG_RAN;
		bool SMS;
		
	};







}















#endif


