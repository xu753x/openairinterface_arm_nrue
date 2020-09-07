#include "PLMNSupportItem.hpp"

extern "C"{
  #include "Ngap_SliceSupportItem.h"
}


#include <iostream>
using namespace std;

namespace ngap{
	PLMNSupportItem::PLMNSupportItem()
	{
		plmn = NULL;
		snssai = NULL;
	}
	PLMNSupportItem::~PLMNSupportItem(){}

	void PLMNSupportItem::setPlmnSliceSupportList(PlmnId * m_plmn, S_NSSAI * m_snssai, int num)
	{
		plmn = m_plmn;
	    snssai = m_snssai;
	    numOfSnssai = num;
	}
	
	bool PLMNSupportItem::encode2PLMNSupportItem(Ngap_PLMNSupportItem_t *plmnsupportItem)
	{
		if(!plmn->encode2octetstring(plmnsupportItem->pLMNIdentity)) return false;
		cout<<"PLMNSupportItem::numOfSnssai	"<<numOfSnssai<<endl;
		for(int i=0;i<numOfSnssai;i++)
		{
			Ngap_SliceSupportItem_t *slice = (Ngap_SliceSupportItem_t*)calloc(1,sizeof(Ngap_SliceSupportItem_t));
			if(!snssai[i].encode2S_NSSAI(&slice->s_NSSAI)) return false;
			ASN_SEQUENCE_ADD(&plmnsupportItem->sliceSupportList.list, slice);
		}
		return true;
	}
	bool PLMNSupportItem::decodefromPLMNSupportItem(Ngap_PLMNSupportItem_t *plmnsupportItem)
	{
		plmn = new PlmnId();
		if(!plmn->decodefromoctetstring(plmnsupportItem->pLMNIdentity)) return false;
		numOfSnssai = plmnsupportItem->sliceSupportList.list.count;
		snssai = new S_NSSAI[numOfSnssai]();
		for(int i=0;i<numOfSnssai;i++)
		{
			if(!snssai[i].decodefromS_NSSAI(&plmnsupportItem->sliceSupportList.list.array[i]->s_NSSAI)) return false;
		}
		return true;
	}
	void PLMNSupportItem::getPlmnSliceSupportList(PlmnId * & m_plmn, S_NSSAI * & m_snssai, int & snssainum)
	{
		m_plmn = plmn;
		m_snssai = snssai;
		snssainum = numOfSnssai;
	}
}
