#ifndef _EUTRACellIdentity_H_
#define _EUTRACellIdentity_H_

extern "C"{
  #include "Ngap_EUTRACellIdentity.h"
}

namespace ngap{

	class EUTRACellIdentity{
	public:
		EUTRACellIdentity();
		virtual ~EUTRACellIdentity();

		bool encode2bitstring(Ngap_EUTRACellIdentity_t&eUTRACellIdentity);
		bool decodefrombitstring(Ngap_EUTRACellIdentity_t &eUTRACellIdentity);
		uint32_t getEUTRACellIdentity();
		void setEUTRACellIdentity(uint32_t m_eutracellidentity);
	private:
		uint32_t eutracellidentity;
	};
}


#endif