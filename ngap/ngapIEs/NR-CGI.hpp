#ifndef _NR_CGI_H_
#define _NR_CGI_H_

#include "PlmnId.hpp"
#include "NRCellIdentity.hpp"

extern "C"{
  #include "Ngap_NR-CGI.h"
}

namespace ngap{

	class NR_CGI{
	public:
		NR_CGI();
		virtual ~NR_CGI();

		void setNR_CGI(PlmnId*,NRCellIdentity*);
		bool encode2NR_CGI(Ngap_NR_CGI_t *);
		bool decodefromNR_CGI(Ngap_NR_CGI_t *);
		void getNR_CGI(PlmnId* &,NRCellIdentity* &);
	private:
		PlmnId *plmnId;
		NRCellIdentity *nRCellIdentity;
	};
}

#endif
