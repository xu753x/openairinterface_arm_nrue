#include "QosFlowSetupRequestList.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	QosFlowSetupRequestList::QosFlowSetupRequestList()
	{
		items = NULL;
		numofitems = 0;
	}
	QosFlowSetupRequestList::~QosFlowSetupRequestList(){}

	void QosFlowSetupRequestList::setQosFlowSetupRequestList(QosFlowSetupRequestItem * m_items, int m_numofitems)
	{
		items = m_items;
		numofitems = m_numofitems;
	}
	bool QosFlowSetupRequestList::getQosFlowSetupRequestList(QosFlowSetupRequestItem * & m_items, int & m_numofitems)
	{
		m_items = items;
		m_numofitems = numofitems;

		if(!items) return false;
		if(!numofitems) return false;

		return true;
	}

	bool QosFlowSetupRequestList::encode2QosFlowSetupRequestList(Ngap_QosFlowSetupRequestList_t *qosFlowSetupRequestList)
	{
		cout<<"QosFlowSetupRequestList::numOfQosFlowSetupRequestItem	"<<numofitems<<endl;
		for(int i=0;i<numofitems;i++)
		{
			Ngap_QosFlowSetupRequestItem_t *item = (Ngap_QosFlowSetupRequestItem_t*)calloc(1,sizeof(Ngap_QosFlowSetupRequestItem_t));
			if(!item) return false;
			if(!items[i].encode2QosFlowSetupRequestItem(item)) return false;
			if(ASN_SEQUENCE_ADD(&qosFlowSetupRequestList->list, item) != 0) return false;
		}
		return true;
	}
	bool QosFlowSetupRequestList::decodefromQosFlowSetupRequestList(Ngap_QosFlowSetupRequestList_t *qosFlowSetupRequestList)
	{
		numofitems = qosFlowSetupRequestList->list.count;
		items = new QosFlowSetupRequestItem[numofitems]();
		for(int i=0;i<numofitems;i++)
		{
			if(!items[i].decodefromQosFlowSetupRequestItem(qosFlowSetupRequestList->list.array[i])) return false;
		}
		return true;
	}

}


