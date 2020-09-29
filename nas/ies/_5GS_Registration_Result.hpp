#ifndef ___5GS_Registration_Result_H_
#define ___5GS_Registration_Result_H_

#include <stdint.h>

namespace nas {

	class _5GS_Registration_Result {
	public:
		_5GS_Registration_Result();
		_5GS_Registration_Result(uint8_t iei);
		_5GS_Registration_Result(const uint8_t iei, bool emergency,bool nssaa, bool sms, uint8_t value);
		~_5GS_Registration_Result();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void setValue(uint8_t value);
		uint8_t getValue();
	private:
		uint8_t _iei;
		bool Emergency;
		bool NSSAA;
		bool SMS;
		uint8_t _value;
	};
}














#endif



