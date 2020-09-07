#ifndef _NRCellIdentity_H_
#define _NRCellIdentity_H_

extern "C"{
  #include "Ngap_NRCellIdentity.h"
}

namespace ngap{

	class NRCellIdentity{
	public:
		NRCellIdentity();
		virtual ~NRCellIdentity();

		bool encode2bitstring(Ngap_NRCellIdentity_t&nRCellIdentity);
		bool decodefrombitstring(Ngap_NRCellIdentity_t &nRCellIdentity);
		unsigned long getNRCellIdentity();
		void setNRCellIdentity(unsigned long m_nrcellidentity);
	private:
		unsigned long nrcellidentity;
	};
}


#endif

