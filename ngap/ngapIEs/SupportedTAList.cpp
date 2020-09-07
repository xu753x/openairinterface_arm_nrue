#include "SupportedTAList.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	SupportedTAList::SupportedTAList()
	{
		supportedTaItem = NULL;
		numberOfSupportedTaItem = 0;
	}
	SupportedTAList::~SupportedTAList(){}

	bool SupportedTAList::encode2SupportedTAList(Ngap_SupportedTAList_t * supportedTAList)
	{
		cout<<"SupportedTAList::numberOfSupportedTaItem	"<<numberOfSupportedTaItem<<endl;
		for(int i=0; i< numberOfSupportedTaItem; i++)
		{
			Ngap_SupportedTAItem_t *ta = (Ngap_SupportedTAItem_t*)calloc(1,sizeof(Ngap_SupportedTAItem_t));
			if(!supportedTaItem[i].encode2SupportedTaItem(ta)) return false;
			if(ASN_SEQUENCE_ADD(&supportedTAList->list, ta) != 0) return false;
		}
		return true;
	}
	bool SupportedTAList::decodefromSupportedTAList(Ngap_SupportedTAList_t *pdu)
	{
		numberOfSupportedTaItem = pdu->list.count;
		supportedTaItem = new SupportedTaItem[numberOfSupportedTaItem]();
		for(int i=0; i< numberOfSupportedTaItem; i++)
		{
			if(!supportedTaItem[i].decodefromSupportedTaItem(pdu->list.array[i])) return false;
		}

		return true;
	}

	void SupportedTAList::setSupportedTaItems(SupportedTaItem * m_supportedTaItem, int numOfItem)
	{
		supportedTaItem = m_supportedTaItem;
		numberOfSupportedTaItem = numOfItem;
	}
	void SupportedTAList::getSupportedTaItems(SupportedTaItem * & m_supportedTaItem, int &numOfItem)
	{
		m_supportedTaItem = supportedTaItem;
		numOfItem = numberOfSupportedTaItem;
	}

}
