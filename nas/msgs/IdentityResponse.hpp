        //itti_msg->cause = 28;//cause nas(2)--deregister
#ifndef _IdentityResponse_H_
#define _IdentityResponse_H_

#include "nas_ie_header.hpp"

namespace nas {

	class IdentityResponse {
	public:
		IdentityResponse();
		~IdentityResponse();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
		void setHeader(uint8_t security_header_type);
		void setSUCI_SUPI_format_IMSI(const string mcc, const string mnc, const string routingInd, uint8_t protection_sch_id, const string msin);
		void setSUCI_SUPI_format_IMSI(const string mcc, const string mnc, const string routingInd, uint8_t protection_sch_id, uint8_t hnpki, const string msin);
		void set5G_GUTI();
		void setIMEI_IMEISV();
		void set5G_S_TMSI();
	  /*** belongs to _5GSMobilityIdentity**/
	public:
		NasMmPlainHeader      *plain_header;
		_5GSMobilityIdentity  *ie_mobility_id;
	};













}



















#endif







