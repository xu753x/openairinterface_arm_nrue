#include "AssociatedQosFlowList.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	AssociatedQosFlowList::AssociatedQosFlowList()
	{
		associatedQosFlowItem = NULL;
		numofassociatedQosFlowItem = 0;
	}  

	AssociatedQosFlowList::~AssociatedQosFlowList(){} 
	
	
	void AssociatedQosFlowList::setAssociatedQosFlowList(AssociatedQosFlowItem *m_associatedQosFlowItem,int numofitem)
	{
		associatedQosFlowItem = m_associatedQosFlowItem;
		numofassociatedQosFlowItem = numofitem;
	}
	bool AssociatedQosFlowList::getAssociatedQosFlowList(AssociatedQosFlowItem *&m_associatedQosFlowItem,int &numofitem)
	{
		m_associatedQosFlowItem = associatedQosFlowItem;
		numofitem = numofassociatedQosFlowItem;
		
		return true;
	}
	
	bool AssociatedQosFlowList::encode2AssociatedQosFlowList(Ngap_AssociatedQosFlowList_t &associatedQosFlowList)
	{
		for(int i=0;i<numofassociatedQosFlowItem;i++)
		{
			Ngap_AssociatedQosFlowItem_t *ie = (Ngap_AssociatedQosFlowItem_t *)calloc(1,sizeof(Ngap_AssociatedQosFlowItem_t));
			if(!ie) return false;
			if(!associatedQosFlowItem[i].encode2AssociatedQosFlowItem(ie)) return false;
			if(ASN_SEQUENCE_ADD(&associatedQosFlowList.list, ie) != 0) return false;
		}

		return true;
	}
	bool AssociatedQosFlowList::decodefromAssociatedQosFlowList(Ngap_AssociatedQosFlowList_t &associatedQosFlowList)
	{
		numofassociatedQosFlowItem = associatedQosFlowList.list.count;
		associatedQosFlowItem = new AssociatedQosFlowItem[numofassociatedQosFlowItem]();
		for(int i=0;i<numofassociatedQosFlowItem;i++)
		{
			if(!associatedQosFlowItem[i].decodefromAssociatedQosFlowItem(associatedQosFlowList.list.array[i])) return false;
		}

		return true;
	}

}


