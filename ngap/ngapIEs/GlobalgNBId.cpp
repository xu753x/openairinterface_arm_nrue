#include "GlobalgNBId.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	GlobalgNBId::GlobalgNBId()
	{
		plmnId = NULL;
		gNB_ID = NULL;
	}  

  	GlobalgNBId::~GlobalgNBId(){} 
  
	void GlobalgNBId::setGlobalgNBId(PlmnId *plmn, GNB_ID *gnbid)
	{
		plmnId = plmn;
		gNB_ID = gnbid;
	}
	bool GlobalgNBId::encode2GlobalgNBId(Ngap_GlobalGNB_ID_t *globalgnbid)
	{
		if(!plmnId->encode2octetstring(globalgnbid->pLMNIdentity)) return false;
		if(!gNB_ID->encode2bitstring(globalgnbid->gNB_ID)) return false;

		return true;
	}
	bool GlobalgNBId::decodefromGlobalgNBId(Ngap_GlobalGNB_ID_t *globalgnbid)
	{
		plmnId = new PlmnId();
		gNB_ID = new GNB_ID();
		if(!plmnId->decodefromoctetstring(globalgnbid->pLMNIdentity)) return false;
		if(!gNB_ID->decodefrombitstring(globalgnbid->gNB_ID)) return false;

		return true;
	}
	void GlobalgNBId::getGlobalgNBId(PlmnId *&plmn, GNB_ID *&gnbid)
	{
		plmn = plmnId;
		gnbid = gNB_ID;
	}
}
