#ifndef _TAI_H_
#define _TAI_H_

#include "PlmnId.hpp"
#include "Tac.hpp"

extern "C"{
  #include "Ngap_TAI.h"
}

namespace ngap{

	class TAI{
	public:
		TAI();
		virtual ~TAI();

		void setTAI(PlmnId*,TAC*);
		bool encode2TAI(Ngap_TAI_t *);
		bool decodefromTAI(Ngap_TAI_t *);
		void getTAI(PlmnId* &,TAC* &);
	private:
		PlmnId *plmnId;
		TAC *tac;
	};
}

#endif