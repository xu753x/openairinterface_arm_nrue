#ifndef __PDU_Session_Reactivation_Result_H_
#define __PDU_Session_Reactivation_Result_H_

#include <stdint.h>

namespace nas {

	class PDU_Session_Reactivation_Result {
	public:
		PDU_Session_Reactivation_Result();
		PDU_Session_Reactivation_Result(uint8_t iei);
		PDU_Session_Reactivation_Result(const uint8_t iei, uint16_t value);
		~PDU_Session_Reactivation_Result();
		void setValue(uint8_t iei, uint16_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		uint16_t getValue();
	private:
		uint8_t _iei;
		uint8_t length;
		uint16_t _value;

	};



}










#endif



