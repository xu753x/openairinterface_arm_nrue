#include "GTP-TEID.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	GtpTeid::GtpTeid(){}  

	GtpTeid::~GtpTeid(){} 
	
	
	void GtpTeid::setGtpTeid(const uint32_t m_gtp_teid)
	{
		gtp_teid = m_gtp_teid;
	}
	bool GtpTeid::getGtpTeid(uint32_t &m_gtp_teid)
	{
		m_gtp_teid = gtp_teid;
		
		return true;
	}
	
	bool GtpTeid::encode2GtpTeid(Ngap_GTP_TEID_t &gtpTeid)
	{
		gtpTeid.size = sizeof(uint32_t);
		gtpTeid.buf = (uint8_t*)calloc(1,sizeof(uint32_t));
		if(!gtpTeid.buf) return false;

		for(int i=0;i<gtpTeid.size;i++)
		{
			gtpTeid.buf[i] = (gtp_teid>>(gtpTeid.size-i-1)*8)&0xff;
		}

		return true;
	}
	bool GtpTeid::decodefromGtpTeid(Ngap_GTP_TEID_t &gtpTeid)
	{
		if(!gtpTeid.buf) return false;

		gtp_teid = 0;
		for(int i=0;i<gtpTeid.size;i++)
		{
			gtp_teid = gtp_teid<<8;
			gtp_teid |= gtpTeid.buf[i];
		}
		
		return true;
	}

}



