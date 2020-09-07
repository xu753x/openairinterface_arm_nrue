#include "AMFName.hpp"

extern "C"{
  #include "OCTET_STRING.h"
}

#include <iostream>
using namespace std;

namespace ngap{

	AmfName::AmfName()
	{
		amfname = NULL;
	}
	AmfName::~AmfName()
	{
		free(amfname);
		amfname = NULL;
	}
	
	void AmfName::setValue(const std::string m_amfName)
	{
		if(amfname)
		{
			free(amfname);
			amfname = NULL;
		}
		amfname = (char *)calloc(1,m_amfName.size()+1);
		memcpy(amfname, m_amfName.c_str(), m_amfName.size());
                amfname[m_amfName.size()] = '\0';
		
	}
	void AmfName::getValue(std::string &m_amfName)
	{
		m_amfName = amfname;
	}
	bool AmfName::encode2AmfName(Ngap_AMFName_t *amfNameIe)
	{   
		if(OCTET_STRING_fromBuf (amfNameIe, amfname, strlen(amfname)) < 0) return false;

		return true;
	}
	bool AmfName::decodefromAmfName(Ngap_AMFName_t * pdu)
	{
		if(!pdu->buf) return false;
		amfname = (char *)pdu->buf;
		return true;
	}

}

