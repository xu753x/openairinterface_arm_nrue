#ifndef  _S_NSSAI_H_
#define  _S_NSSAI_H_

#include <string>

extern "C"{
  #include "Ngap_SST.h"
  #include "Ngap_SD.h"
  #include "Ngap_S-NSSAI.h"
}

namespace ngap{

	class S_NSSAI{
	public:
		S_NSSAI();
		virtual ~S_NSSAI();
		bool sSTEncode2OctetString(Ngap_SST_t&);
		bool sSTdecodefromOctetString(Ngap_SST_t);
		bool sDEncode2OctetString(Ngap_SD_t*);
		bool sDdecodefromOctetString(Ngap_SD_t *);
		void setSst(const std::string charSst);
		void getSst(std::string &charSst);
		void setSd(const std::string charSd);
		bool getSd(std::string & s_nssaiSd);
		bool encode2S_NSSAI(Ngap_S_NSSAI_t *);
		bool decodefromS_NSSAI(Ngap_S_NSSAI_t *);
	private:
		uint8_t  sst;//mandotory  OCTET_STRING(SIZE(1))
		uint32_t sd;//OCTET_STRING(SIZE(3))
		bool  sdIsSet;
	};

}










#endif
