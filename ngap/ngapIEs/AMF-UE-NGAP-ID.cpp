#include "AMF-UE-NGAP-ID.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	AMF_UE_NGAP_ID::AMF_UE_NGAP_ID(){}  

	AMF_UE_NGAP_ID::~AMF_UE_NGAP_ID(){} 

	void AMF_UE_NGAP_ID::setAMF_UE_NGAP_ID(unsigned long m_amfUeNgapId)
	{
		amfUeNgapId = m_amfUeNgapId;
	}
	unsigned long AMF_UE_NGAP_ID::getAMF_UE_NGAP_ID()
	{
		return amfUeNgapId;
	}
	
	bool AMF_UE_NGAP_ID::encode2AMF_UE_NGAP_ID(Ngap_AMF_UE_NGAP_ID_t &amfuengapid)
	{
		amfuengapid.size = 5;
		amfuengapid.buf = (uint8_t *)calloc(1,amfuengapid.size);
		if(!amfuengapid.buf ) return false;

		for(int i=0;i<amfuengapid.size;i++)
		{
			amfuengapid.buf[i] = (amfUeNgapId & (0xff00000000>>i*8))>>((amfuengapid.size-i-1)*8);
		}

		return true;
	}
	bool AMF_UE_NGAP_ID::decodefromAMF_UE_NGAP_ID(Ngap_AMF_UE_NGAP_ID_t &amfuengapid)
	{
		if(!amfuengapid.buf) return false;

		amfUeNgapId = 0;
		for(int i=0;i<amfuengapid.size;i++)
		{
			amfUeNgapId = amfUeNgapId<<8;
			amfUeNgapId |= amfuengapid.buf[i];
		}

		return true;
	}
}


