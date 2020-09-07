#include "PDUSessionResourceSetupListCxtReq.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceSetupListCxtReq::PDUSessionResourceSetupListCxtReq()
	{
		pduSessionResourceSetupItemCxtReq = NULL;
		numofpduSessionResourceSetupItemCxtReq = 0;
	}
	PDUSessionResourceSetupListCxtReq::~PDUSessionResourceSetupListCxtReq(){}
	
	void PDUSessionResourceSetupListCxtReq::setPDUSessionResourceSetupListCxtReq(PDUSessionResourceSetupItemCxtReq *m_pduSessionResourceSetupItemCxtReq,int num)
	{
		pduSessionResourceSetupItemCxtReq = m_pduSessionResourceSetupItemCxtReq;
		numofpduSessionResourceSetupItemCxtReq = num;
	}
	bool PDUSessionResourceSetupListCxtReq::encode2PDUSessionResourceSetupListCxtReq(Ngap_PDUSessionResourceSetupListCxtReq_t *pduSessionResourceSetupListCxtReq)
	{
		for(int i=0;i<numofpduSessionResourceSetupItemCxtReq;i++)
		{
                        cout<<"encoding items"<<endl;
			Ngap_PDUSessionResourceSetupItemCxtReq_t *request = (Ngap_PDUSessionResourceSetupItemCxtReq_t *)calloc(1,sizeof(Ngap_PDUSessionResourceSetupItemCxtReq_t));
			if(!request) return false;
			if(!pduSessionResourceSetupItemCxtReq[i].encode2PDUSessionResourceSetupItemCxtReq(request)) return false;
			if(ASN_SEQUENCE_ADD(&pduSessionResourceSetupListCxtReq->list, request) != 0) return false;
		}
		
		return true; 
	}
	bool PDUSessionResourceSetupListCxtReq::decodefromPDUSessionResourceSetupListCxtReq(Ngap_PDUSessionResourceSetupListCxtReq_t *pduSessionResourceSetupListCxtReq)
	{
		numofpduSessionResourceSetupItemCxtReq = pduSessionResourceSetupListCxtReq->list.count;
		pduSessionResourceSetupItemCxtReq = new PDUSessionResourceSetupItemCxtReq[numofpduSessionResourceSetupItemCxtReq]();
		for(int i=0;i<numofpduSessionResourceSetupItemCxtReq;i++)
		{
			if(!pduSessionResourceSetupItemCxtReq[i].decodefromPDUSessionResourceSetupItemCxtReq(pduSessionResourceSetupListCxtReq->list.array[i])) return false;
		}
		
		return true;
	}
	void PDUSessionResourceSetupListCxtReq::getPDUSessionResourceSetupListCxtReq(PDUSessionResourceSetupItemCxtReq *&m_pduSessionResourceSetupItemCxtReq,int &num)
	{
		m_pduSessionResourceSetupItemCxtReq = pduSessionResourceSetupItemCxtReq;
		num = numofpduSessionResourceSetupItemCxtReq;
	}

}



