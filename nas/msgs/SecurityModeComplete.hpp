#ifndef _SecurityModeComplete_H_
#define _SecurityModeComplete_H_

#include "nas_ie_header.hpp"

namespace nas {

	class SecurityModeComplete {
	public:
		SecurityModeComplete();
		~SecurityModeComplete();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);

		void setIMEISV(IMEISV_t imeisv);
		void setNAS_Message_Container(bstring value);
		void setNON_IMEISV(IMEISV_t imeisv);

		bool getIMEISV(IMEISV_t &imeisv);
		bool getNasMessageContainer(bstring &nas);
		bool getNON_IMEISV(IMEISV_t &imeisv);
	public:
		NasMmPlainHeader       *plain_header;
		_5GSMobilityIdentity   *ie_imeisv;
		NAS_Message_Container  *ie_nas_message_container;
		_5GSMobilityIdentity   *ie_non_imeisvpei;
	};













}



















#endif




