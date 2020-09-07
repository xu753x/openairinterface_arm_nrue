#include "TAI.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	TAI::TAI()
	{
		plmnId = NULL;
		tac = NULL;
	}
	TAI::~TAI(){}

	void TAI::setTAI(PlmnId* m_plmnId,TAC* m_tac)
	{
		plmnId = m_plmnId;
		tac = m_tac;
	}
	bool TAI::encode2TAI(Ngap_TAI_t *tai)
	{
		if(!plmnId->encode2octetstring(tai->pLMNIdentity)) return false;
		if(!tac->encode2octetstring(tai->tAC)) return false;

		return true;
	}
	bool TAI::decodefromTAI(Ngap_TAI_t *tai)
	{
		plmnId = new PlmnId();
		tac = new TAC();
		if(!plmnId->decodefromoctetstring(tai->pLMNIdentity)) return false;
		if(!tac->decodefromoctetstring(tai->tAC)) return false;

		return true;
	}
	void TAI::getTAI(PlmnId* &m_plmnId,TAC* &m_tac)
	{
		m_plmnId = plmnId;
		m_tac = tac;
	}
}
