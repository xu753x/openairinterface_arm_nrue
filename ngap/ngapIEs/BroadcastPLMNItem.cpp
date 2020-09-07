#include "BroadcastPLMNItem.hpp"
#include "S-NSSAI.hpp"
#include "PlmnId.hpp"

extern "C"{
  #include "Ngap_BroadcastPLMNList.h"
  #include "Ngap_SliceSupportItem.h" 
}

#include <iostream>
using namespace std;

namespace ngap{
	BroadcastPLMNItem::BroadcastPLMNItem()
	{
		plmn = NULL;
		snssai = NULL;
		numOfSnssai = 0;
	}
	BroadcastPLMNItem::~BroadcastPLMNItem(){}

	void BroadcastPLMNItem::setPlmnSliceSupportList(PlmnId * m_plmn, S_NSSAI * m_snssai, int num)
	{
		plmn = m_plmn;
		snssai = m_snssai;
		numOfSnssai = num;
	}
	void BroadcastPLMNItem::getPlmnSliceSupportList(PlmnId * & m_plmn, S_NSSAI * & m_snssai, int & snssainum)
	{
		m_plmn = plmn;
		m_snssai = snssai;
		snssainum = numOfSnssai;
	}

	bool BroadcastPLMNItem::encode2BroadcastPLMNItem(Ngap_BroadcastPLMNItem_t * plmnItem)
	{
		if(!plmn->encode2octetstring(plmnItem->pLMNIdentity)) return false;
		cout<<"BroadcastPLMNItem::numOfSnssai	"<<numOfSnssai<<endl;
		for(int i=0;i<numOfSnssai;i++)
		{
			Ngap_SliceSupportItem_t *slice = (Ngap_SliceSupportItem_t*)calloc(1,sizeof(Ngap_SliceSupportItem_t));
			if(!slice) return false;
			if(!snssai[i].encode2S_NSSAI(&slice->s_NSSAI)) return false;
			if(ASN_SEQUENCE_ADD(&plmnItem->tAISliceSupportList.list, slice) != 0) return false;
		}
		return true;
	}
	bool BroadcastPLMNItem::decodefromBroadcastPLMNItem(Ngap_BroadcastPLMNItem_t * pdu)
	{
		plmn = new PlmnId();
		if(!plmn->decodefromoctetstring(pdu->pLMNIdentity)) return false;
		numOfSnssai = pdu->tAISliceSupportList.list.count;
		snssai = new S_NSSAI[numOfSnssai]();
		for(int i=0;i<numOfSnssai;i++)
		{
			if(!snssai[i].decodefromS_NSSAI(&pdu->tAISliceSupportList.list.array[i]->s_NSSAI)) return false;
		}
		return true;
	}

}
