#ifndef _RegistrationComplete_H_
#define _RegistrationComplete_H_

#include "nas_ie_header.hpp"

namespace nas {

	class RegistrationComplete {
	public:
		RegistrationComplete();
		~RegistrationComplete();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);
		void setSOR_Transparent_Container(uint8_t header, uint8_t *value);		
	public:
		NasMmPlainHeader      *plain_header;
		SOR_Transparent_Container *ie_sor_transparent_container;		
	};













}



















#endif






