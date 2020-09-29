#ifndef _Non_3GPP_NW_Provided_Policies_H
#define _Non_3GPP_NW_Provided_Policies_H

#include <stdint.h>

namespace nas {

	class Non_3GPP_NW_Provided_Policies {
	public:
		Non_3GPP_NW_Provided_Policies();
		Non_3GPP_NW_Provided_Policies(const uint8_t iei, uint8_t value);
		~Non_3GPP_NW_Provided_Policies();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void setValue(const uint8_t value);
		uint8_t getValue();
	private:
		uint8_t _iei;	
		uint8_t _value;
	};





}






#endif



