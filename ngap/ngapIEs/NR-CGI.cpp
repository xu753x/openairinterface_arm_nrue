#include "NR-CGI.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	NR_CGI::NR_CGI()
	{
		plmnId = NULL;
		nRCellIdentity = NULL;
	}
	NR_CGI::~NR_CGI(){}

	void NR_CGI::setNR_CGI(PlmnId* m_plmnId,NRCellIdentity*m_nRCellIdentity)
	{
		plmnId = m_plmnId;
		nRCellIdentity = m_nRCellIdentity;
	}
	bool NR_CGI::encode2NR_CGI(Ngap_NR_CGI_t *nr_cgi)
	{
		if(!plmnId->encode2octetstring(nr_cgi->pLMNIdentity)) return false;
		if(!nRCellIdentity->encode2bitstring(nr_cgi->nRCellIdentity)) return false;

		return true;
	}
	bool NR_CGI::decodefromNR_CGI(Ngap_NR_CGI_t *nr_cgi)
	{
		plmnId = new PlmnId();
		nRCellIdentity = new NRCellIdentity();
		if(!plmnId->decodefromoctetstring(nr_cgi->pLMNIdentity)) return false;
		if(!nRCellIdentity->decodefrombitstring(nr_cgi->nRCellIdentity)) return false;

		return true;
	}
	void NR_CGI::getNR_CGI(PlmnId* &m_plmnId,NRCellIdentity* &m_nRCellIdentity)
	{
		m_plmnId = plmnId;
		m_nRCellIdentity = nRCellIdentity;
	}
}

