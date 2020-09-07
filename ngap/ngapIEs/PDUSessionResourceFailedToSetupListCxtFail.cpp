#include "PDUSessionResourceFailedToSetupListCxtFail.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceFailedToSetupListCxtFail::PDUSessionResourceFailedToSetupListCxtFail()
	{
		pduSessionResourceFailedToSetupItemCxtFail = NULL;
		numofpduSessionResourceFailedToSetupItemCxtFail = 0;
	}
	PDUSessionResourceFailedToSetupListCxtFail::~PDUSessionResourceFailedToSetupListCxtFail(){}
	
	void PDUSessionResourceFailedToSetupListCxtFail::setPDUSessionResourceFailedToSetupListCxtFail(PDUSessionResourceFailedToSetupItemCxtFail *m_pduSessionResourceFailedToSetupItemCxtFail,int num)
	{
		pduSessionResourceFailedToSetupItemCxtFail = m_pduSessionResourceFailedToSetupItemCxtFail;
		numofpduSessionResourceFailedToSetupItemCxtFail = num;
	}
	bool PDUSessionResourceFailedToSetupListCxtFail::encode2PDUSessionResourceFailedToSetupListCxtFail(Ngap_PDUSessionResourceFailedToSetupListCxtFail_t *pduSessionResourceFailedToSetupListCxtFail)
	{
		for(int i=0;i<numofpduSessionResourceFailedToSetupItemCxtFail;i++)
		{
			Ngap_PDUSessionResourceFailedToSetupItemCxtFail_t *failedToFailure = (Ngap_PDUSessionResourceFailedToSetupItemCxtFail_t *)calloc(1,sizeof(Ngap_PDUSessionResourceFailedToSetupItemCxtFail_t));
			if(!failedToFailure) return false;
			if(!pduSessionResourceFailedToSetupItemCxtFail[i].encode2PDUSessionResourceFailedToSetupItemCxtFail(failedToFailure)) return false;
			if(ASN_SEQUENCE_ADD(&pduSessionResourceFailedToSetupListCxtFail->list, failedToFailure) != 0) return false;
		}
		
		return true; 
	}
	bool PDUSessionResourceFailedToSetupListCxtFail::decodefromPDUSessionResourceFailedToSetupListCxtFail(Ngap_PDUSessionResourceFailedToSetupListCxtFail_t *pduSessionResourceFailedToSetupListCxtFail)
	{
		numofpduSessionResourceFailedToSetupItemCxtFail = pduSessionResourceFailedToSetupListCxtFail->list.count;
		pduSessionResourceFailedToSetupItemCxtFail = new PDUSessionResourceFailedToSetupItemCxtFail[numofpduSessionResourceFailedToSetupItemCxtFail]();
		for(int i=0;i<numofpduSessionResourceFailedToSetupItemCxtFail;i++)
		{
			if(!pduSessionResourceFailedToSetupItemCxtFail[i].decodefromPDUSessionResourceFailedToSetupItemCxtFail(pduSessionResourceFailedToSetupListCxtFail->list.array[i])) return false;
		}
		
		return true;
	}
	void PDUSessionResourceFailedToSetupListCxtFail::getPDUSessionResourceFailedToSetupListCxtFail(PDUSessionResourceFailedToSetupItemCxtFail *&m_pduSessionResourceFailedToSetupItemCxtFail,int &num)
	{
		m_pduSessionResourceFailedToSetupItemCxtFail = pduSessionResourceFailedToSetupItemCxtFail;
		num = numofpduSessionResourceFailedToSetupItemCxtFail;
	}

}




