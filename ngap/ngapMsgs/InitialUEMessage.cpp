#include "InitialUEMessage.hpp"

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
	InitialUEMessageMsg::InitialUEMessageMsg()
	{
		initialUEMessagePdu = NULL;
		initialUEMessageIEs = NULL;
		ranUeNgapId = NULL;
		nasPdu = NULL;
		userLocationInformation = NULL;
		rRCEstablishmentCause = NULL;
		uEContextRequest = NULL;
                fivegSTmsi = NULL;
	}
	InitialUEMessageMsg::~InitialUEMessageMsg(){}

	void InitialUEMessageMsg::setMessageType()
	{
		if(!initialUEMessagePdu) initialUEMessagePdu = (Ngap_NGAP_PDU_t*)calloc(1, sizeof(Ngap_NGAP_PDU_t));

		MessageType InitialUEMessageMessageTypeIE;
		InitialUEMessageMessageTypeIE.setProcedureCode(Ngap_ProcedureCode_id_InitialUEMessage);
		InitialUEMessageMessageTypeIE.setTypeOfMessage(Ngap_NGAP_PDU_PR_initiatingMessage);
		InitialUEMessageMessageTypeIE.setCriticality(Ngap_Criticality_ignore);
		InitialUEMessageMessageTypeIE.setValuePresent(Ngap_InitiatingMessage__value_PR_InitialUEMessage);
		
		if(InitialUEMessageMessageTypeIE.getProcedureCode()==Ngap_ProcedureCode_id_InitialUEMessage && InitialUEMessageMessageTypeIE.getTypeOfMessage()==Ngap_NGAP_PDU_PR_initiatingMessage && InitialUEMessageMessageTypeIE.getCriticality()==Ngap_Criticality_ignore)
		{
			InitialUEMessageMessageTypeIE.encode2pdu(initialUEMessagePdu);
			initialUEMessageIEs = &(initialUEMessagePdu->choice.initiatingMessage->value.choice.InitialUEMessage);
		}
		else
		{
			cout<<"[warning] This information doesn't refer to InitialUEMessage Message!!!"<<endl;
		}
	}
	void InitialUEMessageMsg::setRanUENgapID(uint32_t ran_ue_ngap_id)
	{
		if(!ranUeNgapId)
			ranUeNgapId = new RAN_UE_NGAP_ID();
		ranUeNgapId->setRanUeNgapId(ran_ue_ngap_id);

		Ngap_InitialUEMessage_IEs_t *ie = (Ngap_InitialUEMessage_IEs_t *)calloc(1,sizeof(Ngap_InitialUEMessage_IEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_RAN_UE_NGAP_ID;
		ie->criticality = Ngap_Criticality_reject;
		ie->value.present = Ngap_InitialUEMessage_IEs__value_PR_RAN_UE_NGAP_ID;
		
		int ret = ranUeNgapId->encode2RAN_UE_NGAP_ID(ie->value.choice.RAN_UE_NGAP_ID);
		if(!ret)
		{
			cout<<"encode RAN_UE_NGAP_ID IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&initialUEMessageIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode RAN_UE_NGAP_ID IE error"<<endl;
	}
	void InitialUEMessageMsg::setNasPdu(uint8_t *nas,size_t sizeofnas)
	{
		if(!nasPdu)
			nasPdu = new NAS_PDU();

		nasPdu->setNasPdu(nas,sizeofnas);
		
		Ngap_InitialUEMessage_IEs_t *ie = (Ngap_InitialUEMessage_IEs_t *)calloc(1,sizeof(Ngap_InitialUEMessage_IEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_NAS_PDU;
		ie->criticality = Ngap_Criticality_reject;
		ie->value.present = Ngap_InitialUEMessage_IEs__value_PR_NAS_PDU;
		
		int ret = nasPdu->encode2octetstring(ie->value.choice.NAS_PDU);
		if(!ret)
		{
			cout<<"encode NAS_PDU IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&initialUEMessageIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode NAS_PDU IE error"<<endl;
	}
	void InitialUEMessageMsg::setUserLocationInfoNR(struct NrCgi_s cig, struct Tai_s tai)
	{
		if(!userLocationInformation)
			userLocationInformation = new UserLocationInformation();

		//userLocationInformation->setInformation(UserLocationInformationEUTRA * informationEUTRA);
		UserLocationInformationNR *informationNR = new UserLocationInformationNR();
		NR_CGI *nR_CGI = new NR_CGI();
		PlmnId *plmnId_cgi = new PlmnId();
		NRCellIdentity *nRCellIdentity = new NRCellIdentity();
		plmnId_cgi->setMccMnc(cig.mcc,cig.mnc);
		nRCellIdentity->setNRCellIdentity(cig.nrCellID);
		nR_CGI->setNR_CGI(plmnId_cgi,nRCellIdentity);

		TAI *tai_nr = new TAI();
		PlmnId *plmnId_tai = new PlmnId();
		plmnId_tai->setMccMnc(tai.mcc,tai.mnc);
		TAC *tac = new TAC();
		tac->setTac(tai.tac);
		tai_nr->setTAI(plmnId_tai, tac);
		informationNR->setInformationNR(nR_CGI,tai_nr);
		userLocationInformation->setInformation(informationNR);
		
		Ngap_InitialUEMessage_IEs_t *ie = (Ngap_InitialUEMessage_IEs_t *)calloc(1,sizeof(Ngap_InitialUEMessage_IEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_UserLocationInformation;
		ie->criticality = Ngap_Criticality_reject;
		ie->value.present = Ngap_InitialUEMessage_IEs__value_PR_UserLocationInformation;
		
		int ret = userLocationInformation->encodefromUserLocationInformation(&ie->value.choice.UserLocationInformation);
		if(!ret)
		{
			cout<<"encode UserLocationInformation IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&initialUEMessageIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode UserLocationInformation IE error"<<endl;
	}
	void InitialUEMessageMsg::setRRCEstablishmentCause(e_Ngap_RRCEstablishmentCause cause_value)
	{
		if(!rRCEstablishmentCause)
			rRCEstablishmentCause = new RRCEstablishmentCause();
		
		rRCEstablishmentCause->setRRCEstablishmentCause(cause_value);

		Ngap_InitialUEMessage_IEs_t *ie = (Ngap_InitialUEMessage_IEs_t *)calloc(1,sizeof(Ngap_InitialUEMessage_IEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_RRCEstablishmentCause;
		ie->criticality = Ngap_Criticality_ignore;
		ie->value.present = Ngap_InitialUEMessage_IEs__value_PR_RRCEstablishmentCause;
		
		int ret = rRCEstablishmentCause->encode2RRCEstablishmentCause(ie->value.choice.RRCEstablishmentCause);
		if(!ret)
		{
			cout<<"encode RRCEstablishmentCause IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&initialUEMessageIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode RRCEstablishmentCause IE error"<<endl;
	}
	//void InitialUEMessageMsg::set5GS_TMSI(string amfSetId, string amfPointer, string _5g_tmsi);
	void InitialUEMessageMsg::setUeContextRequest(e_Ngap_UEContextRequest ueCtxReq)
	{		
		if(!uEContextRequest)
			uEContextRequest = new UEContextRequest();
		
		uEContextRequest->setUEContextRequest(ueCtxReq);

		Ngap_InitialUEMessage_IEs_t *ie = (Ngap_InitialUEMessage_IEs_t *)calloc(1,sizeof(Ngap_InitialUEMessage_IEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_UEContextRequest;
		ie->criticality = Ngap_Criticality_ignore;
		ie->value.present = Ngap_InitialUEMessage_IEs__value_PR_UEContextRequest;
		
		int ret = uEContextRequest->encode2UEContextRequest(ie->value.choice.UEContextRequest);
		if(!ret)
		{
			cout<<"encode UEContextRequest IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&initialUEMessageIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode UEContextRequest IE error"<<endl;
	}
	int  InitialUEMessageMsg::encode2buffer(uint8_t *buf, int buf_size)
	{
		asn_fprint(stderr, &asn_DEF_Ngap_NGAP_PDU, initialUEMessagePdu);
		asn_enc_rval_t er = aper_encode_to_buffer(&asn_DEF_Ngap_NGAP_PDU, NULL, initialUEMessagePdu, buf, buf_size);
    	cout<<"er.encoded("<<er.encoded<<")"<<endl;
		return er.encoded;
	}
	//Decapsulation
	bool InitialUEMessageMsg::decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu)
	{
  		initialUEMessagePdu = ngap_msg_pdu;
		
	    if(initialUEMessagePdu->present == Ngap_NGAP_PDU_PR_initiatingMessage)
		{
                  //cout<<"test0515: procedureCode    "<<initialUEMessagePdu->choice.initiatingMessage->procedureCode<<endl;
                  //cout<<"test0515: criticality    "<<initialUEMessagePdu->choice.initiatingMessage->criticality<<endl;
                  //cout<<"test0515: present    "<<initialUEMessagePdu->choice.initiatingMessage->value.present<<endl;
			if(initialUEMessagePdu->choice.initiatingMessage && initialUEMessagePdu->choice.initiatingMessage->procedureCode == Ngap_ProcedureCode_id_InitialUEMessage && initialUEMessagePdu->choice.initiatingMessage->criticality == Ngap_Criticality_ignore && initialUEMessagePdu->choice.initiatingMessage->value.present == Ngap_InitiatingMessage__value_PR_InitialUEMessage)
			{
				initialUEMessageIEs = &initialUEMessagePdu->choice.initiatingMessage->value.choice.InitialUEMessage;
			}
			else
			{
				cout<<"Check InitialUEMessage message error!!!"<<endl;
                return false;
			}
		}
		else
		{
			cout<<"MessageType error!!!"<<endl;
            return false;
		}
		for(int i=0; i< initialUEMessageIEs->protocolIEs.list.count; i++)
		{
			switch(initialUEMessageIEs->protocolIEs.list.array[i]->id)
			{
				case Ngap_ProtocolIE_ID_id_RAN_UE_NGAP_ID:{
					if(initialUEMessageIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_reject && initialUEMessageIEs->protocolIEs.list.array[i]->value.present == Ngap_InitialUEMessage_IEs__value_PR_RAN_UE_NGAP_ID)
					{
						ranUeNgapId = new RAN_UE_NGAP_ID();
						if(!ranUeNgapId->decodefromRAN_UE_NGAP_ID(initialUEMessageIEs->protocolIEs.list.array[i]->value.choice.RAN_UE_NGAP_ID))
						{
							cout<<"decoded ngap RAN_UE_NGAP_ID IE error"<<endl;
							return false;
						}
                                                cout<<"in initialUeMessage, received ranUeNgapId "<<ranUeNgapId->getRanUeNgapId()<<endl;
					}
					else
					{
						cout<<"decoded ngap RAN_UE_NGAP_ID IE error"<<endl;
						return false;
					}
				}break;
				case Ngap_ProtocolIE_ID_id_NAS_PDU:{
					if(initialUEMessageIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_reject && initialUEMessageIEs->protocolIEs.list.array[i]->value.present == Ngap_InitialUEMessage_IEs__value_PR_NAS_PDU)
					{
						nasPdu = new NAS_PDU();
						if(!nasPdu->decodefromoctetstring(initialUEMessageIEs->protocolIEs.list.array[i]->value.choice.NAS_PDU))
						{
							cout<<"decoded ngap NAS_PDU IE error"<<endl;
							return false;
						}
					}
					else
					{
						cout<<"decoded ngap NAS_PDU IE error"<<endl;
						return false;
					}
				}break;
				case Ngap_ProtocolIE_ID_id_UserLocationInformation:{
					if(initialUEMessageIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_reject && initialUEMessageIEs->protocolIEs.list.array[i]->value.present == Ngap_InitialUEMessage_IEs__value_PR_UserLocationInformation)
					{
						userLocationInformation = new UserLocationInformation();
						if(!userLocationInformation->decodefromUserLocationInformation(&initialUEMessageIEs->protocolIEs.list.array[i]->value.choice.UserLocationInformation))
						{
							cout<<"decoded ngap UserLocationInformation IE error"<<endl;
							return false;
						}
					}
					else
					{
						cout<<"decoded ngap UserLocationInformation IE error"<<endl;
						return false;
					}
				}break;
				case Ngap_ProtocolIE_ID_id_RRCEstablishmentCause:{
					if(initialUEMessageIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_ignore && initialUEMessageIEs->protocolIEs.list.array[i]->value.present == Ngap_InitialUEMessage_IEs__value_PR_RRCEstablishmentCause)
					{
						rRCEstablishmentCause = new RRCEstablishmentCause();
						if(!rRCEstablishmentCause->decodefromRRCEstablishmentCause(initialUEMessageIEs->protocolIEs.list.array[i]->value.choice.RRCEstablishmentCause))
						{
							cout<<"decoded ngap RRCEstablishmentCause IE error"<<endl;
							return false;
						}
					}
					else
					{
						cout<<"decoded ngap RRCEstablishmentCause IE error"<<endl;
						return false;
					}
				}break;
				case Ngap_ProtocolIE_ID_id_UEContextRequest:{
					if(initialUEMessageIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_ignore && initialUEMessageIEs->protocolIEs.list.array[i]->value.present == Ngap_InitialUEMessage_IEs__value_PR_UEContextRequest)
					{
						uEContextRequest = new UEContextRequest();
						if(!uEContextRequest->decodefromUEContextRequest(initialUEMessageIEs->protocolIEs.list.array[i]->value.choice.UEContextRequest))
						{
							cout<<"decoded ngap UEContextRequest IE error"<<endl;
							return false;
						}
					}
					else
					{
						cout<<"decoded ngap UEContextRequest IE error"<<endl;
						return false;
					}
				}break;
			
                                case Ngap_ProtocolIE_ID_id_FiveG_S_TMSI:{
				  if(initialUEMessageIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_reject && initialUEMessageIEs->protocolIEs.list.array[i]->value.present == Ngap_InitialUEMessage_IEs__value_PR_FiveG_S_TMSI){
                                    fivegSTmsi = new FiveGSTmsi();
                                    if(!fivegSTmsi->decodefrompdu(initialUEMessageIEs->protocolIEs.list.array[i]->value.choice.FiveG_S_TMSI)){
                                      cout <<"decode ngap FiveG_S_TMSI IE error"<<endl;
                                      return false;
                                    }
                                  }
                                }break; 
	
				default:{
					cout<<"not decoded IE:"<<initialUEMessageIEs->protocolIEs.list.array[i]->id<<endl; 
					return true;
				}
			}
		}

		return true;
  	}
	uint32_t InitialUEMessageMsg::getRanUENgapID()
	{
		return ranUeNgapId->getRanUeNgapId();
	}
	bool InitialUEMessageMsg::getNasPdu(uint8_t *&nas,size_t &sizeofnas)
	{		
		if(!nasPdu->getNasPdu(nas,sizeofnas)) return false;
		
		return true;
	}
	bool InitialUEMessageMsg::getUserLocationInfoNR(struct NrCgi_s &cig, struct Tai_s &tai)
	{
		UserLocationInformationNR *informationNR;
		userLocationInformation->getInformation(informationNR);
		if(userLocationInformation->getChoiceOfUserLocationInformation() != Ngap_UserLocationInformation_PR_userLocationInformationNR) return false;
		NR_CGI *nR_CGI;
		TAI *nR_TAI;
		informationNR->getInformationNR(nR_CGI, nR_TAI);
		PlmnId *cgi_plmnId;
		NRCellIdentity *nRCellIdentity;
		nR_CGI->getNR_CGI(cgi_plmnId, nRCellIdentity);
		cgi_plmnId->getMcc(cig.mcc);
		cgi_plmnId->getMnc(cig.mnc);
		cig.nrCellID = nRCellIdentity->getNRCellIdentity();

		PlmnId *tai_plmnId;
		TAC *tac;
		nR_TAI->getTAI(tai_plmnId, tac);
		tai_plmnId->getMcc(tai.mcc);
		tai_plmnId->getMnc(tai.mnc);
		tai.tac = tac->getTac();

		return true;
	}
	int  InitialUEMessageMsg::getRRCEstablishmentCause()
	{
		return rRCEstablishmentCause->getRRCEstablishmentCause();
	}
	int  InitialUEMessageMsg::getUeContextRequest()
	{       cout<<"testing 11.1"<<endl;
		return uEContextRequest->getUEContextRequest();
	}

bool InitialUEMessageMsg::get5GS_TMSI(string &_5g_s_tmsi){
  if(fivegSTmsi){
    fivegSTmsi->getValue(_5g_s_tmsi);
    return true;
  }else return false;
}

}

