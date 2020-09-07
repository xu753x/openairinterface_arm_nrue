#include "RAN-UE-NGAP-ID.hpp"
//#include "String2Value.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	RAN_UE_NGAP_ID::RAN_UE_NGAP_ID(){}  

	RAN_UE_NGAP_ID::~RAN_UE_NGAP_ID(){} 

	void RAN_UE_NGAP_ID::setRanUeNgapId(uint32_t m_ranuengapid)
	{
		ranUeNgapId = m_ranuengapid;
	}
	uint32_t RAN_UE_NGAP_ID::getRanUeNgapId()
	{
		return ranUeNgapId;
	}
	
	bool RAN_UE_NGAP_ID::encode2RAN_UE_NGAP_ID(Ngap_RAN_UE_NGAP_ID_t &ranuengapid)
	{
		ranuengapid = ranUeNgapId;
		return true;
	}
	bool RAN_UE_NGAP_ID::decodefromRAN_UE_NGAP_ID(Ngap_RAN_UE_NGAP_ID_t &ranuengapid)
	{
		ranUeNgapId = ranuengapid;
                cout<<"in libngap, received ranuengapid "<<ranUeNgapId<<endl;
		return true;
	}
}


