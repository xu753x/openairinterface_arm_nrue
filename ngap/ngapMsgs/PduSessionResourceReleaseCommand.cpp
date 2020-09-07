#include "PduSessionResourceReleaseCommand.hpp"

extern "C"{
  #include "constr_TYPE.h"
  #include "asn_codecs.h"
  #include "per_encoder.h"
  #include "per_decoder.h"
  #include "constraints.h"
}

#include <iostream>
using namespace std;

namespace ngap{
	PduSessionResourceReleaseCommand::PduSessionResourceReleaseCommand()
	{
		pduSessionResourceReleaseCommandPdu = NULL;
		pduSessionResourceReleaseCommandIEs = NULL;
		amfUeNgapId = NULL;
		ranUeNgapId = NULL;
		pduSessionResourceReleaseCommandList = NULL;
	}
	PduSessionResourceReleaseCommand::~PduSessionResourceReleaseCommand(){}

	void PduSessionResourceReleaseCommand::setMessageType()
	{
		if(!pduSessionResourceReleaseCommandPdu) pduSessionResourceReleaseCommandPdu = (Ngap_NGAP_PDU_t*)calloc(1, sizeof(Ngap_NGAP_PDU_t));

		MessageType pduSessionResourceToReleasePduTypeIE;
		pduSessionResourceToReleasePduTypeIE.setProcedureCode(Ngap_ProcedureCode_id_PDUSessionResourceRelease);
		pduSessionResourceToReleasePduTypeIE.setTypeOfMessage(Ngap_NGAP_PDU_PR_initiatingMessage);
		pduSessionResourceToReleasePduTypeIE.setCriticality(Ngap_Criticality_reject);
		pduSessionResourceToReleasePduTypeIE.setValuePresent(Ngap_InitiatingMessage__value_PR_PDUSessionResourceReleaseCommand);
		
		if(pduSessionResourceToReleasePduTypeIE.getProcedureCode()== Ngap_ProcedureCode_id_PDUSessionResourceRelease && pduSessionResourceToReleasePduTypeIE.getTypeOfMessage()== Ngap_NGAP_PDU_PR_initiatingMessage && pduSessionResourceToReleasePduTypeIE.getCriticality()==Ngap_Criticality_reject)
		{
			pduSessionResourceToReleasePduTypeIE.encode2pdu(pduSessionResourceReleaseCommandPdu);
			pduSessionResourceReleaseCommandIEs = &(pduSessionResourceReleaseCommandPdu->choice.initiatingMessage->value.choice.PDUSessionResourceReleaseCommand);
		}
		else
		{
			cout<<"[warning] This information doesn't refer to PDUSessionResourceSetupResponse Message!!!"<<endl;
		}
	}
	void PduSessionResourceReleaseCommand::setAmfUeNgapId(unsigned long id)
	{
		if(!amfUeNgapId)
			amfUeNgapId = new AMF_UE_NGAP_ID();
		amfUeNgapId->setAMF_UE_NGAP_ID(id);
		
		Ngap_PDUSessionResourceReleaseCommandIEs_t *ie = (Ngap_PDUSessionResourceReleaseCommandIEs_t *)calloc(1,sizeof(Ngap_PDUSessionResourceReleaseCommandIEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_AMF_UE_NGAP_ID;
		ie->criticality = Ngap_Criticality_ignore;
		ie->value.present = Ngap_PDUSessionResourceReleaseCommandIEs__value_PR_AMF_UE_NGAP_ID;
		
		int ret = amfUeNgapId->encode2AMF_UE_NGAP_ID(ie->value.choice.AMF_UE_NGAP_ID);
		if(!ret)
		{
			cout<<"encode AMF_UE_NGAP_ID IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&pduSessionResourceReleaseCommandIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode AMF_UE_NGAP_ID IE error"<<endl;
	}
	void PduSessionResourceReleaseCommand::setRanUeNgapId(uint32_t ran_ue_ngap_id)
	{
		if(!ranUeNgapId)
			ranUeNgapId = new RAN_UE_NGAP_ID();
		ranUeNgapId->setRanUeNgapId(ran_ue_ngap_id);

		Ngap_PDUSessionResourceReleaseCommandIEs_t *ie = (Ngap_PDUSessionResourceReleaseCommandIEs_t *)calloc(1,sizeof(Ngap_PDUSessionResourceReleaseCommandIEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_RAN_UE_NGAP_ID;
		ie->criticality = Ngap_Criticality_ignore;
		ie->value.present = Ngap_PDUSessionResourceReleaseCommandIEs__value_PR_RAN_UE_NGAP_ID;
		
		int ret = ranUeNgapId->encode2RAN_UE_NGAP_ID(ie->value.choice.RAN_UE_NGAP_ID);
		if(!ret)
		{
			cout<<"encode RAN_UE_NGAP_ID IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&pduSessionResourceReleaseCommandIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode RAN_UE_NGAP_ID IE error"<<endl;
	}

	void PduSessionResourceReleaseCommand::setPduSessionResourceToReleaseList(std::vector<PDUSessionResourceReleaseCommandItem_t>list)
	{
		if(!pduSessionResourceReleaseCommandList)
			pduSessionResourceReleaseCommandList = new PDUSessionResourceReleaseListCmd();

		
		PDUSessionResourceReleaseItemCmd * m_pduSessionResourceReleaseItemRelCmd = new PDUSessionResourceReleaseItemCmd[list.size()]();

		for(int i=0;i<list.size();i++)
		{
			PDUSessionID * m_pDUSessionID = new PDUSessionID();
			m_pDUSessionID->setPDUSessionID(list[i].pduSessionId);
			
			m_pduSessionResourceReleaseItemRelCmd[i].setPDUSessionResourceReleaseItemCmd(m_pDUSessionID,list[i].pduSessionResourceReleaseCommandTransfer);
		}
		
		pduSessionResourceReleaseCommandList->setPDUSessionResourceReleaseListCmd(m_pduSessionResourceReleaseItemRelCmd, list.size());
		
		Ngap_PDUSessionResourceReleaseCommandIEs_t *ie = (Ngap_PDUSessionResourceReleaseCommandIEs_t *)calloc(1,sizeof(Ngap_PDUSessionResourceReleaseCommandIEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_PDUSessionResourceToReleaseListRelCmd;
		ie->criticality = Ngap_Criticality_reject;
		ie->value.present = Ngap_PDUSessionResourceReleaseCommandIEs__value_PR_PDUSessionResourceToReleaseListRelCmd;
		
		int ret = pduSessionResourceReleaseCommandList->encode2PDUSessionResourceReleaseListCmd(&ie->value.choice.PDUSessionResourceToReleaseListRelCmd);
		if(!ret)
		{
			cout<<"encode PDUSessionResourceToReleaseListRelCmd IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&pduSessionResourceReleaseCommandIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode PDUSessionResourceToReleaseListRelCmd IE error"<<endl;
		
	}
	
	int  PduSessionResourceReleaseCommand::encode2buffer(uint8_t *buf, int buf_size)
	{
		asn_fprint(stderr, &asn_DEF_Ngap_NGAP_PDU, pduSessionResourceReleaseCommandPdu);
		asn_enc_rval_t er = aper_encode_to_buffer(&asn_DEF_Ngap_NGAP_PDU, NULL, pduSessionResourceReleaseCommandPdu, buf, buf_size);
    	cout<<"er.encoded("<<er.encoded<<")"<<endl;
		return er.encoded;
	}
	//Decapsulation
	/*bool PduSessionResourceReleaseCommand::decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu)
	{
  		pduSessionResourceSetupResponsePdu = ngap_msg_pdu;
		//asn_fprint(stderr, &asn_DEF_Ngap_NGAP_PDU, pduSessionResourceSetupResponsePdu);
		
	    if(pduSessionResourceSetupResponsePdu->present == Ngap_NGAP_PDU_PR_successfulOutcome)
		{
			if(pduSessionResourceSetupResponsePdu->choice.successfulOutcome && pduSessionResourceSetupResponsePdu->choice.successfulOutcome->procedureCode == Ngap_ProcedureCode_id_PDUSessionResourceSetup && pduSessionResourceSetupResponsePdu->choice.successfulOutcome->criticality == Ngap_Criticality_reject && pduSessionResourceSetupResponsePdu->choice.successfulOutcome->value.present == Ngap_SuccessfulOutcome__value_PR_PDUSessionResourceSetupResponse)
			{
				pduSessionResourceSetupResponseIEs = &pduSessionResourceSetupResponsePdu->choice.successfulOutcome->value.choice.PDUSessionResourceSetupResponse;
			}
			else
			{
				cout<<"Check PDUSessionResourceSetupResponse message error!!!"<<endl;
                return false;
			}
		}
		else
		{
			cout<<"MessageType error!!!"<<endl;
            return false;
		}
		for(int i=0; i< pduSessionResourceSetupResponseIEs->protocolIEs.list.count; i++)
		{
			switch(pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->id)
			{
				case Ngap_ProtocolIE_ID_id_AMF_UE_NGAP_ID:{
					if(pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_ignore && pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->value.present == Ngap_PDUSessionResourceSetupResponseIEs__value_PR_AMF_UE_NGAP_ID)
					{
						amfUeNgapId = new AMF_UE_NGAP_ID();
						if(!amfUeNgapId->decodefromAMF_UE_NGAP_ID(pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->value.choice.AMF_UE_NGAP_ID))
						{
							cout<<"decoded ngap AMF_UE_NGAP_ID IE error"<<endl;
							return false;
						}
					}
					else
					{
						cout<<"decoded ngap AMF_UE_NGAP_ID IE error"<<endl;
						return false;
					}
				}break;
				case Ngap_ProtocolIE_ID_id_RAN_UE_NGAP_ID:{
					if(pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_ignore && pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->value.present == Ngap_PDUSessionResourceSetupResponseIEs__value_PR_RAN_UE_NGAP_ID)
					{
						ranUeNgapId = new RAN_UE_NGAP_ID();
						if(!ranUeNgapId->decodefromRAN_UE_NGAP_ID(pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->value.choice.RAN_UE_NGAP_ID))
						{
							cout<<"decoded ngap RAN_UE_NGAP_ID IE error"<<endl;
							return false;
						}
					}
					else
					{
						cout<<"decoded ngap RAN_UE_NGAP_ID IE error"<<endl;
						return false;
					}
				}break;
				case Ngap_ProtocolIE_ID_id_PDUSessionResourceSetupListSURes:{
					if(pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_ignore && pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->value.present == Ngap_PDUSessionResourceSetupResponseIEs__value_PR_PDUSessionResourceSetupListSURes)
					{
						pduSessionResourceSetupResponseList = new PDUSessionResourceSetupListSURes();
						if(!pduSessionResourceSetupResponseList->decodefromPDUSessionResourceSetupListSURes(&pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->value.choice.PDUSessionResourceSetupListSURes))
						{
							cout<<"decoded ngap PDUSessionResourceSetupListSURes IE error"<<endl;
							return false;
						}
					}
					else
					{
						cout<<"decoded ngap PDUSessionResourceSetupListSURes IE error"<<endl;
						return false;
					}
				}break;
				case Ngap_ProtocolIE_ID_id_PDUSessionResourceFailedToSetupListSURes:{
					if(pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_ignore && pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->value.present == Ngap_PDUSessionResourceSetupResponseIEs__value_PR_PDUSessionResourceFailedToSetupListSURes)
					{
						pduSessionResourceFailedToSetupResponseList = new PDUSessionResourceFailedToSetupListSURes();
						if(!pduSessionResourceFailedToSetupResponseList->decodefromPDUSessionResourceFailedToSetupListSURes(&pduSessionResourceSetupResponseIEs->protocolIEs.list.array[i]->value.choice.PDUSessionResourceFailedToSetupListSURes))
						{
							cout<<"decoded ngap PDUSessionResourceFailedToSetupListSURes IE error"<<endl;
							return false;
						}
					}
					else
					{
						cout<<"decoded ngap PDUSessionResourceFailedToSetupListSURes IE error!"<<endl;
						return false;
					}
				}break;
				
				default:{
					cout<<"decoded ngap message pdu error"<<endl; 
					return false;
				}
			}
		}

		return true;
  	}*/
	unsigned long PduSessionResourceReleaseCommand::getAmfUeNgapId()
	{
		if(!amfUeNgapId) return -1;
		return amfUeNgapId->getAMF_UE_NGAP_ID();
	}
	uint32_t PduSessionResourceReleaseCommand::getRanUeNgapId()
	{
		if(!ranUeNgapId) return -1;
		return ranUeNgapId->getRanUeNgapId();
	}
	/*bool PduSessionResourceReleaseCommand::getPduSessionResourceSetupResponseList(std::vector<PDUSessionResourceSetupResponseItem_t> &list)
	{
		if(!pduSessionResourceSetupResponseList) return false;
		
		PDUSessionResourceSetupItemSURes *m_pduSessionResourceSetupItemSURes;
		int num = 0;
		pduSessionResourceSetupResponseList->getPDUSessionResourceSetupListSURes(m_pduSessionResourceSetupItemSURes,num);

		for(int i=0;i<num;i++)
		{
			PDUSessionResourceSetupResponseItem_t response;
			
			PDUSessionID *m_pDUSessionID;
			m_pduSessionResourceSetupItemSURes[i].getPDUSessionResourceSetupItemSURes(m_pDUSessionID,response.pduSessionResourceSetupResponseTransfer);
			m_pDUSessionID->getPDUSessionID(response.pduSessionId);			
			
			list.push_back(response);
		}
		
		return true;
	}
	*/
}




