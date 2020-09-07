#include "PDUSessionResourceSetupListCxtRes.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceSetupListCxtRes::PDUSessionResourceSetupListCxtRes()
	{
		pduSessionResourceSetupItemCxtRes = NULL;
		numofpduSessionResourceSetupItemCxtRes = 0;
	}
	PDUSessionResourceSetupListCxtRes::~PDUSessionResourceSetupListCxtRes(){}
	
	void PDUSessionResourceSetupListCxtRes::setPDUSessionResourceSetupListCxtRes(PDUSessionResourceSetupItemCxtRes *m_pduSessionResourceSetupItemCxtRes,int num)
	{
		pduSessionResourceSetupItemCxtRes = m_pduSessionResourceSetupItemCxtRes;
		numofpduSessionResourceSetupItemCxtRes = num;
	}
	bool PDUSessionResourceSetupListCxtRes::encode2PDUSessionResourceSetupListCxtRes(Ngap_PDUSessionResourceSetupListCxtRes_t *pduSessionResourceSetupListCxtRes)
	{
		for(int i=0;i<numofpduSessionResourceSetupItemCxtRes;i++)
		{
			Ngap_PDUSessionResourceSetupItemCxtRes_t *response = (Ngap_PDUSessionResourceSetupItemCxtRes_t *)calloc(1,sizeof(Ngap_PDUSessionResourceSetupItemCxtRes_t));
			if(!response) return false;
			if(!pduSessionResourceSetupItemCxtRes[i].encode2PDUSessionResourceSetupItemCxtRes(response)) return false;
			if(ASN_SEQUENCE_ADD(&pduSessionResourceSetupListCxtRes->list, response) != 0) return false;
		}
		
		return true; 
	}
	bool PDUSessionResourceSetupListCxtRes::decodefromPDUSessionResourceSetupListCxtRes(Ngap_PDUSessionResourceSetupListCxtRes_t *pduSessionResourceSetupListCxtRes)
	{
		numofpduSessionResourceSetupItemCxtRes = pduSessionResourceSetupListCxtRes->list.count;
		pduSessionResourceSetupItemCxtRes = new PDUSessionResourceSetupItemCxtRes[numofpduSessionResourceSetupItemCxtRes]();
		for(int i=0;i<numofpduSessionResourceSetupItemCxtRes;i++)
		{
			if(!pduSessionResourceSetupItemCxtRes[i].decodefromPDUSessionResourceSetupItemCxtRes(pduSessionResourceSetupListCxtRes->list.array[i])) return false;
		}
		
		return true;
	}
	void PDUSessionResourceSetupListCxtRes::getPDUSessionResourceSetupListCxtRes(PDUSessionResourceSetupItemCxtRes *&m_pduSessionResourceSetupItemCxtRes,int &num)
	{
		m_pduSessionResourceSetupItemCxtRes = pduSessionResourceSetupItemCxtRes;
		num = numofpduSessionResourceSetupItemCxtRes;
	}

}




