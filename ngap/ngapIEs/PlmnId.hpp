#ifndef _PlmnId_H
#define _PlmnId_H

extern "C"{
  #include "Ngap_PLMNIdentity.h"
}

#include <string>

namespace ngap{

	class PlmnId{
	public:
		PlmnId();
		virtual ~PlmnId();
		
		bool encode2octetstring(Ngap_PLMNIdentity_t&);
		bool decodefromoctetstring(Ngap_PLMNIdentity_t&);
		void setMccMnc(const std::string mcc, const std::string mnc);
		void getMcc(std::string &mcc);
		void getMnc(std::string &mnc);
	private:
		uint8_t mcc_digit1;
		uint8_t mcc_digit2;
		uint8_t mcc_digit3;
		uint8_t mnc_digit1;// in case of 2 digit MNC, it should be 0xf
		uint8_t mnc_digit2;
		uint8_t mnc_digit3;
	};

}




#endif

