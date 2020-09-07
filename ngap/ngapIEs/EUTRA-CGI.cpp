#include "EUTRA-CGI.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	EUTRA_CGI::EUTRA_CGI()
	{
		plmnId = NULL;
		eUTRACellIdentity = NULL;
	}
	EUTRA_CGI::~EUTRA_CGI(){}

	void EUTRA_CGI::setEUTRA_CGI(PlmnId* m_plmnId,EUTRACellIdentity* m_eUTRACellIdentity)
	{
		plmnId = m_plmnId;
		eUTRACellIdentity = m_eUTRACellIdentity;
	}
	bool EUTRA_CGI::encode2EUTRA_CGI(Ngap_EUTRA_CGI_t *eutra_cgi)
	{
		if(!plmnId->encode2octetstring(eutra_cgi->pLMNIdentity)) return false;
		if(!eUTRACellIdentity->encode2bitstring(eutra_cgi->eUTRACellIdentity)) return false;

		return true;
	}
	bool EUTRA_CGI::decodefromEUTRA_CGI(Ngap_EUTRA_CGI_t *eutra_cgi)
	{
		plmnId = new PlmnId();
		eUTRACellIdentity = new EUTRACellIdentity();
		if(!plmnId->decodefromoctetstring(eutra_cgi->pLMNIdentity)) return false;
		if(!eUTRACellIdentity->decodefrombitstring(eutra_cgi->eUTRACellIdentity)) return false;

		return true;
	}
	void EUTRA_CGI::getEUTRA_CGI(PlmnId* &m_plmnId,EUTRACellIdentity* &m_eUTRACellIdentity)
	{
		m_plmnId = plmnId;
		m_eUTRACellIdentity = eUTRACellIdentity;
	}
}
