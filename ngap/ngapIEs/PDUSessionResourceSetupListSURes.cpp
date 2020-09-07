#include "PDUSessionResourceSetupListSURes.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceSetupListSURes::PDUSessionResourceSetupListSURes()
	{
		pduSessionResourceSetupItemSURes = NULL;
		numofpduSessionResourceSetupItemSURes = 0;
	}
	PDUSessionResourceSetupListSURes::~PDUSessionResourceSetupListSURes(){}
	
	void PDUSessionResourceSetupListSURes::setPDUSessionResourceSetupListSURes(PDUSessionResourceSetupItemSURes *m_pduSessionResourceSetupItemSURes,int num)
	{
		pduSessionResourceSetupItemSURes = m_pduSessionResourceSetupItemSURes;
		numofpduSessionResourceSetupItemSURes = num;
	}
	bool PDUSessionResourceSetupListSURes::encode2PDUSessionResourceSetupListSURes(Ngap_PDUSessionResourceSetupListSURes_t *pduSessionResourceSetupListSURes)
	{
		for(int i=0;i<numofpduSessionResourceSetupItemSURes;i++)
		{
			Ngap_PDUSessionResourceSetupItemSURes_t *response = (Ngap_PDUSessionResourceSetupItemSURes_t *)calloc(1,sizeof(Ngap_PDUSessionResourceSetupItemSURes_t));
			if(!response) return false;
			if(!pduSessionResourceSetupItemSURes[i].encode2PDUSessionResourceSetupItemSURes(response)) return false;
			if(ASN_SEQUENCE_ADD(&pduSessionResourceSetupListSURes->list, response) != 0) return false;
		}
		
		return true; 
	}
	bool PDUSessionResourceSetupListSURes::decodefromPDUSessionResourceSetupListSURes(Ngap_PDUSessionResourceSetupListSURes_t *pduSessionResourceSetupListSURes)
	{
		numofpduSessionResourceSetupItemSURes = pduSessionResourceSetupListSURes->list.count;
		pduSessionResourceSetupItemSURes = new PDUSessionResourceSetupItemSURes[numofpduSessionResourceSetupItemSURes]();
		for(int i=0;i<numofpduSessionResourceSetupItemSURes;i++)
		{
			if(!pduSessionResourceSetupItemSURes[i].decodefromPDUSessionResourceSetupItemSURes(pduSessionResourceSetupListSURes->list.array[i])) return false;
		}
		
		return true;
	}
	void PDUSessionResourceSetupListSURes::getPDUSessionResourceSetupListSURes(PDUSessionResourceSetupItemSURes *&m_pduSessionResourceSetupItemSURes,int &num)
	{
		m_pduSessionResourceSetupItemSURes = pduSessionResourceSetupItemSURes;
		num = numofpduSessionResourceSetupItemSURes;
	}

}




