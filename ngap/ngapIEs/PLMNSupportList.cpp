#include "PLMNSupportList.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	PLMNSupportList::PLMNSupportList()
	{
		plmnsupportItemItem = NULL;
	}
	PLMNSupportList::~PLMNSupportList(){}

	void PLMNSupportList::addPLMNSupportItems(PLMNSupportItem * m_plmnsupportItemItem, int numOfItem)
	{
		plmnsupportItemItem = m_plmnsupportItemItem;
		numberOfplmnsupportItemItem = numOfItem;
	}
  	bool PLMNSupportList::encode2PLMNSupportList(Ngap_PLMNSupportList_t *plmnsupportList)
	{
	    cout<<"PLMNSupportList::numberOfplmnsupportItemItem	("<<numberOfplmnsupportItemItem<<")"<<endl;
	    for(int i=0; i< numberOfplmnsupportItemItem; i++)
		{
			Ngap_PLMNSupportItem_t *supportItem = (Ngap_PLMNSupportItem_t*)calloc(1,sizeof(Ngap_PLMNSupportItem_t));
			if(!supportItem) return false;
			if(!plmnsupportItemItem[i].encode2PLMNSupportItem(supportItem)) return false;
			if(ASN_SEQUENCE_ADD(&plmnsupportList->list, supportItem) != 0) return false;
	    }
		return true;
  	}
	bool PLMNSupportList::decodefromPLMNSupportList(Ngap_PLMNSupportList_t *plmnsupportList)
	{
		numberOfplmnsupportItemItem = plmnsupportList->list.count;
		plmnsupportItemItem = new PLMNSupportItem[numberOfplmnsupportItemItem]();
		for(int i=0; i< numberOfplmnsupportItemItem; i++)
		{
			if(!plmnsupportItemItem[i].decodefromPLMNSupportItem(plmnsupportList->list.array[i])) return false;
		}

		return true;
	}

	void PLMNSupportList::getPLMNSupportItems(PLMNSupportItem * & m_plmnsupportItemItem, int &numOfItem)
	{
		m_plmnsupportItemItem = plmnsupportItemItem;
		numOfItem = numberOfplmnsupportItemItem;
	}

}

