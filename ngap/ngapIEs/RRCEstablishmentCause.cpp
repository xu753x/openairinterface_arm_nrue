#include "RRCEstablishmentCause.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	RRCEstablishmentCause::RRCEstablishmentCause()
	{
		rrcestablishmentcause = -1;
	}  

	RRCEstablishmentCause::~RRCEstablishmentCause(){} 

	void RRCEstablishmentCause::setRRCEstablishmentCause(e_Ngap_RRCEstablishmentCause m_rrcestablishmentcause)
	{
		rrcestablishmentcause = m_rrcestablishmentcause;
	}
	int RRCEstablishmentCause::getRRCEstablishmentCause()
	{
		return rrcestablishmentcause;
	}
	
	bool RRCEstablishmentCause::encode2RRCEstablishmentCause(Ngap_RRCEstablishmentCause_t &rRCEstablishmentCause)
	{
		rRCEstablishmentCause = rrcestablishmentcause;
		return true;
	}
	bool RRCEstablishmentCause::decodefromRRCEstablishmentCause(Ngap_RRCEstablishmentCause_t &rRCEstablishmentCause)
	{
		rrcestablishmentcause = rRCEstablishmentCause;
		return true;
	}
}



