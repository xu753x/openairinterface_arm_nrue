#ifndef __PDU_Session_Reactivation_Result_Error_Cause_H_
#define __PDU_Session_Reactivation_Result_Error_Cause_H_

#include <stdint.h>

namespace nas {

	class PDU_Session_Reactivation_Result_Error_Cause {
	public:
		PDU_Session_Reactivation_Result_Error_Cause();
		PDU_Session_Reactivation_Result_Error_Cause(uint8_t iei);
		PDU_Session_Reactivation_Result_Error_Cause(const uint8_t iei,uint8_t session_id, uint8_t value);
		~PDU_Session_Reactivation_Result_Error_Cause();
		void setValue(uint8_t session_id, uint8_t value);
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		uint8_t getValue();
	private:
		uint8_t _iei;
		uint16_t length;
		uint8_t SESSION_ID;
		uint8_t _value;

	};



}










#endif



