#ifndef _RRCESTABLISHMENTCAUSE_H_
#define _RRCESTABLISHMENTCAUSE_H_

extern "C"{
  #include "Ngap_RRCEstablishmentCause.h"
}

namespace ngap{

	class RRCEstablishmentCause{
	public:
		RRCEstablishmentCause();
		virtual ~RRCEstablishmentCause();
		
		void setRRCEstablishmentCause(e_Ngap_RRCEstablishmentCause m_rrcestablishmentcause);
		int getRRCEstablishmentCause();
		
		bool encode2RRCEstablishmentCause(Ngap_RRCEstablishmentCause_t &);
		bool decodefromRRCEstablishmentCause(Ngap_RRCEstablishmentCause_t &);
	private:
		int rrcestablishmentcause;
};

}

#endif


