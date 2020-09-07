#include "PDUSessionResourceSetupListSUReq.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceSetupListSUReq::PDUSessionResourceSetupListSUReq()
	{
		pduSessionResourceSetupItemSUReq = NULL;
		numofpduSessionResourceSetupItemSUReq = 0;
	}
	PDUSessionResourceSetupListSUReq::~PDUSessionResourceSetupListSUReq(){}
	
	void PDUSessionResourceSetupListSUReq::setPDUSessionResourceSetupListSUReq(PDUSessionResourceSetupItemSUReq *m_pduSessionResourceSetupItemSUReq,int num)
	{
		pduSessionResourceSetupItemSUReq = m_pduSessionResourceSetupItemSUReq;
		numofpduSessionResourceSetupItemSUReq = num;
	}
	bool PDUSessionResourceSetupListSUReq::encode2PDUSessionResourceSetupListSUReq(Ngap_PDUSessionResourceSetupListSUReq_t *pduSessionResourceSetupListSUReq)
	{
		for(int i=0;i<numofpduSessionResourceSetupItemSUReq;i++)
		{
			Ngap_PDUSessionResourceSetupItemSUReq_t *request = (Ngap_PDUSessionResourceSetupItemSUReq_t *)calloc(1,sizeof(Ngap_PDUSessionResourceSetupItemSUReq_t));
			if(!request) return false;
			if(!pduSessionResourceSetupItemSUReq[i].encode2PDUSessionResourceSetupItemSUReq(request)) return false;
			if(ASN_SEQUENCE_ADD(&pduSessionResourceSetupListSUReq->list, request) != 0) return false;
		}
		
		return true; 
	}
	bool PDUSessionResourceSetupListSUReq::decodefromPDUSessionResourceSetupListSUReq(Ngap_PDUSessionResourceSetupListSUReq_t *pduSessionResourceSetupListSUReq)
	{
		numofpduSessionResourceSetupItemSUReq = pduSessionResourceSetupListSUReq->list.count;
		pduSessionResourceSetupItemSUReq = new PDUSessionResourceSetupItemSUReq[numofpduSessionResourceSetupItemSUReq]();
		for(int i=0;i<numofpduSessionResourceSetupItemSUReq;i++)
		{
			if(!pduSessionResourceSetupItemSUReq[i].decodefromPDUSessionResourceSetupItemSUReq(pduSessionResourceSetupListSUReq->list.array[i])) return false;
		}
		
		return true;
	}
	void PDUSessionResourceSetupListSUReq::getPDUSessionResourceSetupListSUReq(PDUSessionResourceSetupItemSUReq *&m_pduSessionResourceSetupItemSUReq,int &num)
	{
		m_pduSessionResourceSetupItemSUReq = pduSessionResourceSetupItemSUReq;
		num = numofpduSessionResourceSetupItemSUReq;
	}

}



