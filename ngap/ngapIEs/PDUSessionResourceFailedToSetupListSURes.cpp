#include "PDUSessionResourceFailedToSetupListSURes.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceFailedToSetupListSURes::PDUSessionResourceFailedToSetupListSURes()
	{
		pduSessionResourceFailedToSetupItemSURes = NULL;
		numofpduSessionResourceFailedToSetupItemSURes = 0;
	}
	PDUSessionResourceFailedToSetupListSURes::~PDUSessionResourceFailedToSetupListSURes(){}
	
	void PDUSessionResourceFailedToSetupListSURes::setPDUSessionResourceFailedToSetupListSURes(PDUSessionResourceFailedToSetupItemSURes *m_pduSessionResourceFailedToSetupItemSURes,int num)
	{
		pduSessionResourceFailedToSetupItemSURes = m_pduSessionResourceFailedToSetupItemSURes;
		numofpduSessionResourceFailedToSetupItemSURes = num;
	}
	bool PDUSessionResourceFailedToSetupListSURes::encode2PDUSessionResourceFailedToSetupListSURes(Ngap_PDUSessionResourceFailedToSetupListSURes_t *pduSessionResourceFailedToSetupListSURes)
	{
		for(int i=0;i<numofpduSessionResourceFailedToSetupItemSURes;i++)
		{
			Ngap_PDUSessionResourceFailedToSetupItemSURes_t *failedToResponse = (Ngap_PDUSessionResourceFailedToSetupItemSURes_t *)calloc(1,sizeof(Ngap_PDUSessionResourceFailedToSetupItemSURes_t));
			if(!failedToResponse) return false;
			if(!pduSessionResourceFailedToSetupItemSURes[i].encode2PDUSessionResourceFailedToSetupItemSURes(failedToResponse)) return false;
			if(ASN_SEQUENCE_ADD(&pduSessionResourceFailedToSetupListSURes->list, failedToResponse) != 0) return false;
		}
		
		return true; 
	}
	bool PDUSessionResourceFailedToSetupListSURes::decodefromPDUSessionResourceFailedToSetupListSURes(Ngap_PDUSessionResourceFailedToSetupListSURes_t *pduSessionResourceFailedToSetupListSURes)
	{
		numofpduSessionResourceFailedToSetupItemSURes = pduSessionResourceFailedToSetupListSURes->list.count;
		pduSessionResourceFailedToSetupItemSURes = new PDUSessionResourceFailedToSetupItemSURes[numofpduSessionResourceFailedToSetupItemSURes]();
		for(int i=0;i<numofpduSessionResourceFailedToSetupItemSURes;i++)
		{
			if(!pduSessionResourceFailedToSetupItemSURes[i].decodefromPDUSessionResourceFailedToSetupItemSURes(pduSessionResourceFailedToSetupListSURes->list.array[i])) return false;
		}
		
		return true;
	}
	void PDUSessionResourceFailedToSetupListSURes::getPDUSessionResourceFailedToSetupListSURes(PDUSessionResourceFailedToSetupItemSURes *&m_pduSessionResourceFailedToSetupItemSURes,int &num)
	{
		m_pduSessionResourceFailedToSetupItemSURes = pduSessionResourceFailedToSetupItemSURes;
		num = numofpduSessionResourceFailedToSetupItemSURes;
	}

}




