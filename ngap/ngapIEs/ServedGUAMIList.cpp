#include "ServedGUAMIList.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	ServedGUAMIList::ServedGUAMIList()
	{
		servedGUAMIItem = NULL;
	}
	ServedGUAMIList::~ServedGUAMIList(){}

	void ServedGUAMIList::addServedGUAMIItems(ServedGUAMIItem * m_supportedGuamiItem, int numOfItem)
	{
		servedGUAMIItem = m_supportedGuamiItem;
		numberOfservedGUAMIItem = numOfItem;
	}
  	bool ServedGUAMIList::encode2ServedGUAMIList(Ngap_ServedGUAMIList_t *servedGUAMIList)
	{
	    cout<<"ServedGUAMIList::numberOfservedGUAMIItem	("<<numberOfservedGUAMIItem<<")"<<endl;
	    for(int i=0; i< numberOfservedGUAMIItem; i++)
		{
			Ngap_ServedGUAMIItem *guamiItem = (Ngap_ServedGUAMIItem *)calloc(1,sizeof(Ngap_ServedGUAMIItem));
			if(!guamiItem) return false;
			if(!servedGUAMIItem[i].encode2ServedGUAMIItem(guamiItem)) return false;
			if(ASN_SEQUENCE_ADD(&servedGUAMIList->list, guamiItem) != 0) return false;
	    }
		return true;
  	}
	bool ServedGUAMIList::decodefromServedGUAMIList(Ngap_ServedGUAMIList_t *pdu)
	{
		numberOfservedGUAMIItem = pdu->list.count;
		servedGUAMIItem = new ServedGUAMIItem[numberOfservedGUAMIItem]();
		for(int i=0; i< numberOfservedGUAMIItem; i++)
		{
			if(!servedGUAMIItem[i].decodefromServedGUAMIItem(pdu->list.array[i])) return false;
		}
		return true;
	}

	void ServedGUAMIList::getServedGUAMIItems(ServedGUAMIItem * & m_supportedGuamiItem, int &numOfItem)
	{
		m_supportedGuamiItem = servedGUAMIItem;
		numOfItem = numberOfservedGUAMIItem;
	}

}

