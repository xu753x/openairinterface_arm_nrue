#include "SupportedTaItem.hpp"
#include "String2Value.hpp"
#include "BroadcastPLMNItem.hpp"

extern "C"{
  #include "Ngap_SupportedTAItem.h"
  #include "Ngap_BroadcastPLMNItem.h"
}

#include <string>
using namespace std;

namespace ngap{
	SupportedTaItem::SupportedTaItem()
	{
		tac = NULL;
		broadcastPLMNItem = NULL;
		numberOfBroadcastItem = 0;
	}
	SupportedTaItem::~SupportedTaItem(){}

	void SupportedTaItem::setTac(TAC *m_tac)
	{
		tac = m_tac;
	}
	void SupportedTaItem::getTac(TAC *&m_tac)
	{
		m_tac = tac;
	}

	void SupportedTaItem::setBroadcastPlmnList(BroadcastPLMNItem * m_broadcastPLMNItem, int numOfItem)
	{
		numberOfBroadcastItem = numOfItem;
		broadcastPLMNItem = m_broadcastPLMNItem;
	}
	void SupportedTaItem::getBroadcastPlmnList(BroadcastPLMNItem * & m_broadcastPLMNItem, int & numOfItem)
	{
		numOfItem = numberOfBroadcastItem;
		m_broadcastPLMNItem = broadcastPLMNItem;
	}
	bool SupportedTaItem::encode2SupportedTaItem(Ngap_SupportedTAItem_t *ta)
	{
		if(!tac->encode2octetstring(ta->tAC)) return false;
		cout<<"SupportedTaItem::numberOfBroadcastItem	"<<numberOfBroadcastItem<<endl;
		for(int i=0;i<numberOfBroadcastItem;i++)
		{
			Ngap_BroadcastPLMNItem_t *plmnItem = (Ngap_BroadcastPLMNItem*)calloc(1,sizeof(Ngap_BroadcastPLMNItem));
			if(!broadcastPLMNItem[i].encode2BroadcastPLMNItem(plmnItem)) return false;
			if(ASN_SEQUENCE_ADD(&ta->broadcastPLMNList.list, plmnItem) != 0) return false;
		}
		return true;
	}
	bool SupportedTaItem::decodefromSupportedTaItem(Ngap_SupportedTAItem_t *pdu)
	{
		tac = new TAC();
		if(!tac->decodefromoctetstring(pdu->tAC)) return false;
		numberOfBroadcastItem = pdu->broadcastPLMNList.list.count;
		broadcastPLMNItem = new BroadcastPLMNItem[numberOfBroadcastItem]();
		for(int i=0;i<numberOfBroadcastItem;i++)
		{
			if(!broadcastPLMNItem[i].decodefromBroadcastPLMNItem(pdu->broadcastPLMNList.list.array[i])) return false;
		}
		return true;
	}

}
