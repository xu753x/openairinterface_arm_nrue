#include "PDUSessionResourceReleaseListCmd.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	PDUSessionResourceReleaseListCmd::PDUSessionResourceReleaseListCmd()
	{
		pduSessionResourceReleaseItemCmd = NULL;
		numofpduSessionResourceReleaseItemCmd = 0;
	}
	PDUSessionResourceReleaseListCmd::~PDUSessionResourceReleaseListCmd(){}
	
	void PDUSessionResourceReleaseListCmd::setPDUSessionResourceReleaseListCmd(PDUSessionResourceReleaseItemCmd *m_pduSessionResourceReleaseItemCmd,int num)
	{
		pduSessionResourceReleaseItemCmd = m_pduSessionResourceReleaseItemCmd;
		numofpduSessionResourceReleaseItemCmd = num;
	}
	bool PDUSessionResourceReleaseListCmd::encode2PDUSessionResourceReleaseListCmd(Ngap_PDUSessionResourceToReleaseListRelCmd_t *pduSessionResourceReleaseListCmd)
	{
		for(int i=0;i< numofpduSessionResourceReleaseItemCmd;i++)
		{
			Ngap_PDUSessionResourceToReleaseItemRelCmd_t *response = (Ngap_PDUSessionResourceToReleaseItemRelCmd_t *)calloc(1,sizeof(Ngap_PDUSessionResourceToReleaseItemRelCmd_t));
			if(!response) return false;
			if(!pduSessionResourceReleaseItemCmd[i].encode2PDUSessionResourceReleaseItemCmd(response)) return false;
			if(ASN_SEQUENCE_ADD(&pduSessionResourceReleaseListCmd->list, response) != 0) return false;
		}
		
		return true; 
	}
	bool PDUSessionResourceReleaseListCmd::decodefromPDUSessionResourceReleaseListCmd(Ngap_PDUSessionResourceToReleaseListRelCmd_t *pduSessionResourceReleaseListCmd)
	{
		numofpduSessionResourceReleaseItemCmd = pduSessionResourceReleaseListCmd->list.count;
		pduSessionResourceReleaseItemCmd = new PDUSessionResourceReleaseItemCmd[numofpduSessionResourceReleaseItemCmd]();
		for(int i=0;i< numofpduSessionResourceReleaseItemCmd;i++)
		{
			if(!pduSessionResourceReleaseItemCmd[i].decodefromPDUSessionResourceReleaseItemCmd(pduSessionResourceReleaseListCmd->list.array[i])) return false;
		}
		
		return true;
	}
	void PDUSessionResourceReleaseListCmd::getPDUSessionResourceReleaseListCmd(PDUSessionResourceReleaseItemCmd *&m_pduSessionResourceReleaseItemCmd,int &num)
	{
		m_pduSessionResourceReleaseItemCmd = pduSessionResourceReleaseItemCmd;
		num = numofpduSessionResourceReleaseItemCmd;
	}

}




