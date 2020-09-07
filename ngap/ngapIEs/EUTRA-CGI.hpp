#ifndef _EUTRA_CGI_H_
#define _EUTRA_CGI_H_

#include "PlmnId.hpp"
#include "EUTRACellIdentity.hpp"

extern "C"{
  #include "Ngap_EUTRA-CGI.h"
}

namespace ngap{

	class EUTRA_CGI{
	public:
		EUTRA_CGI();
		virtual ~EUTRA_CGI();

		void setEUTRA_CGI(PlmnId*,EUTRACellIdentity*);
		bool encode2EUTRA_CGI(Ngap_EUTRA_CGI_t *);
		bool decodefromEUTRA_CGI(Ngap_EUTRA_CGI_t *);
		void getEUTRA_CGI(PlmnId* &,EUTRACellIdentity* &);
	private:
		PlmnId *plmnId;
		EUTRACellIdentity *eUTRACellIdentity;
	};
}


#endif
