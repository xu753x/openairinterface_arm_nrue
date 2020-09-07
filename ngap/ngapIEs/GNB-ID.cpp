#include "GNB-ID.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	GNB_ID::GNB_ID(){}

  	GNB_ID::~GNB_ID(){} 

	void GNB_ID::setValue(uint32_t gnbId)
	{
		gNBId.id = gnbId;
		uint8_t len = 0;
		for(uint32_t i = 0x00000001; i<= 0x00000400; i=i<<1,len++)
		{
			if((i & gnbId))
			{
				gNBId.biteslen = 32 - len;
				break;
			}
		}
		if(!((gNBId.biteslen >= 22)&&(gNBId.biteslen <= 32)))
		{
			cout<<"[warning][gNBID length out of range]"<<endl;
		}
	}
	bool GNB_ID::encode2bitstring(Ngap_GNB_ID_t &gnbid)
	{
		gnbid.present = Ngap_GNB_ID_PR_gNB_ID;
		
		if(!(gNBId.biteslen % 8))
			gnbid.choice.gNB_ID.size = gNBId.biteslen / 8;
		else
			gnbid.choice.gNB_ID.size = gNBId.biteslen / 8 + 1;
		
		//printf("m_gNBId.size(%d)\n",m_gNBId.size);
		gnbid.choice.gNB_ID.bits_unused = 32 - gNBId.biteslen;
		gnbid.choice.gNB_ID.buf = (uint8_t*)calloc(1,4*sizeof(uint8_t));
		if(!gnbid.choice.gNB_ID.buf) return false;
		gnbid.choice.gNB_ID.buf[3] = gNBId.id & 0x000000ff;
		gnbid.choice.gNB_ID.buf[2] = (gNBId.id & 0x0000ff00)>>8;
		gnbid.choice.gNB_ID.buf[1] = (gNBId.id & 0x00ff0000)>>16;
		gnbid.choice.gNB_ID.buf[0] = (gNBId.id & 0xff000000)>>24;

		return true;
	}
	bool GNB_ID::decodefrombitstring(Ngap_GNB_ID_t &gnbid)
	{
		if(gnbid.present != Ngap_GNB_ID_PR_gNB_ID) return false;
		if(!gnbid.choice.gNB_ID.buf) return false;
		
		gNBId.id = gnbid.choice.gNB_ID.buf[0]<<24;
		gNBId.id |= gnbid.choice.gNB_ID.buf[1]<<16;
		gNBId.id |= gnbid.choice.gNB_ID.buf[2]<<8;
		gNBId.id |= gnbid.choice.gNB_ID.buf[3];
		
		return true;
	}
	long GNB_ID::getValue()
	{
		return gNBId.id;
	}
	
}

