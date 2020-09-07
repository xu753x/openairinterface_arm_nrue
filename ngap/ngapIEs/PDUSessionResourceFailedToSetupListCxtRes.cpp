#include "PDUSessionResourceFailedToSetupListCxtRes.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceFailedToSetupListCxtRes::PDUSessionResourceFailedToSetupListCxtRes()
	{
		pduSessionResourceFailedToSetupItemCxtRes = NULL;
		numofpduSessionResourceFailedToSetupItemCxtRes = 0;
	}
	PDUSessionResourceFailedToSetupListCxtRes::~PDUSessionResourceFailedToSetupListCxtRes(){}
	
	void PDUSessionResourceFailedToSetupListCxtRes::setPDUSessionResourceFailedToSetupListCxtRes(PDUSessionResourceFailedToSetupItemCxtRes *m_pduSessionResourceFailedToSetupItemCxtRes,int num)
	{
		pduSessionResourceFailedToSetupItemCxtRes = m_pduSessionResourceFailedToSetupItemCxtRes;
		numofpduSessionResourceFailedToSetupItemCxtRes = num;
	}
	bool PDUSessionResourceFailedToSetupListCxtRes::encode2PDUSessionResourceFailedToSetupListCxtRes(Ngap_PDUSessionResourceFailedToSetupListCxtRes_t *pduSessionResourceFailedToSetupListCxtRes)
	{
		for(int i=0;i<numofpduSessionResourceFailedToSetupItemCxtRes;i++)
		{
			Ngap_PDUSessionResourceFailedToSetupItemCxtRes_t *failedToResponse = (Ngap_PDUSessionResourceFailedToSetupItemCxtRes_t *)calloc(1,sizeof(Ngap_PDUSessionResourceFailedToSetupItemCxtRes_t));
			if(!failedToResponse) return false;
			if(!pduSessionResourceFailedToSetupItemCxtRes[i].encode2PDUSessionResourceFailedToSetupItemCxtRes(failedToResponse)) return false;
			if(ASN_SEQUENCE_ADD(&pduSessionResourceFailedToSetupListCxtRes->list, failedToResponse) != 0) return false;
		}
		
		return true; 
	}
	bool PDUSessionResourceFailedToSetupListCxtRes::decodefromPDUSessionResourceFailedToSetupListCxtRes(Ngap_PDUSessionResourceFailedToSetupListCxtRes_t *pduSessionResourceFailedToSetupListCxtRes)
	{
		numofpduSessionResourceFailedToSetupItemCxtRes = pduSessionResourceFailedToSetupListCxtRes->list.count;
		pduSessionResourceFailedToSetupItemCxtRes = new PDUSessionResourceFailedToSetupItemCxtRes[numofpduSessionResourceFailedToSetupItemCxtRes]();
		for(int i=0;i<numofpduSessionResourceFailedToSetupItemCxtRes;i++)
		{
			if(!pduSessionResourceFailedToSetupItemCxtRes[i].decodefromPDUSessionResourceFailedToSetupItemCxtRes(pduSessionResourceFailedToSetupListCxtRes->list.array[i])) return false;
		}
		
		return true;
	}
	void PDUSessionResourceFailedToSetupListCxtRes::getPDUSessionResourceFailedToSetupListCxtRes(PDUSessionResourceFailedToSetupItemCxtRes *&m_pduSessionResourceFailedToSetupItemCxtRes,int &num)
	{
		m_pduSessionResourceFailedToSetupItemCxtRes = pduSessionResourceFailedToSetupItemCxtRes;
		num = numofpduSessionResourceFailedToSetupItemCxtRes;
	}

}




