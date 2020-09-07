#include "GlobalRanNodeId.hpp"


#include <iostream>
using namespace std;

namespace ngap{

	GlobalRanNodeId::GlobalRanNodeId()
	{
		globalgNBId = NULL;
	} 

	GlobalRanNodeId::~GlobalRanNodeId(){}

	void GlobalRanNodeId::setChoiceOfRanNodeId(Ngap_GlobalRANNodeID_PR m_idPresent)
	{
		idPresent = m_idPresent;
	}
	Ngap_GlobalRANNodeID_PR GlobalRanNodeId::getChoiceOfRanNodeId()
	{
		return idPresent;
	}

	void GlobalRanNodeId::setGlobalgNBID(GlobalgNBId *m_globalgNBId)
	{
		globalgNBId = m_globalgNBId;
	}
	void GlobalRanNodeId::getGlobalgNBID(GlobalgNBId* & ptr)
	{
		ptr = globalgNBId;
	}

	bool GlobalRanNodeId::encode2GlobalRANNodeID(Ngap_GlobalRANNodeID_t * globalRANNodeID)
	{
		globalRANNodeID->present = idPresent;  

		switch(idPresent)
		{
			case Ngap_GlobalRANNodeID_PR_globalGNB_ID:{
				Ngap_GlobalGNB_ID_t *globalGNB_ID = (Ngap_GlobalGNB_ID_t*)calloc(1, sizeof(struct Ngap_GlobalGNB_ID));
				if(!globalGNB_ID) return false;
				if(!globalgNBId->encode2GlobalgNBId(globalGNB_ID)) return false;
				globalRANNodeID->choice.globalGNB_ID = globalGNB_ID;
			break;}
			case Ngap_GlobalRANNodeID_PR_globalNgENB_ID:{

			break;}
			case Ngap_GlobalRANNodeID_PR_globalN3IWF_ID:{

			break;}
			default:
				cout<<"[Warning] GlobalRanNodeId Present encode error!"<<endl;
				return false;
		}
		return true;
	}
	bool GlobalRanNodeId::decodefromGlobalRANNodeID(Ngap_GlobalRANNodeID_t * globalRANNodeID)
	{
		idPresent = globalRANNodeID->present;
		switch(idPresent)
		{
			case Ngap_GlobalRANNodeID_PR_globalGNB_ID:{
				globalgNBId = new GlobalgNBId();
				if(!globalgNBId->decodefromGlobalgNBId(globalRANNodeID->choice.globalGNB_ID)) return false;
			break;}
			case Ngap_GlobalRANNodeID_PR_globalNgENB_ID:{
				cout<<"[Warning] GlobalRANNodeID Present is globalNgENB!"<<endl;
			break;}
			case Ngap_GlobalRANNodeID_PR_globalN3IWF_ID:{
				cout<<"[Warning] GlobalRanNodeId Present is globalN3IWF!"<<endl;
			break;}
			default:
				cout<<"[Warning] GlobalRanNodeId Present decode error!"<<endl;
				return false;
		}
		return true;
	}
}
