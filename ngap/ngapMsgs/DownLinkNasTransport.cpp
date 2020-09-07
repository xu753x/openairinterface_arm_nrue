#include "DownLinkNasTransport.hpp"

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
	DownLinkNasTransportMsg::DownLinkNasTransportMsg()
	{
		downLinkNasTransportPdu = NULL;
		downLinkNasTransportIEs = NULL;
		amfUeNgapId = NULL;
		ranUeNgapId = NULL;
		oldAmfName = NULL;
		ranPagingPriority = NULL;
		nasPdu = NULL;
		indexToRFSP = NULL;
	}
	DownLinkNasTransportMsg::~DownLinkNasTransportMsg(){}

	void DownLinkNasTransportMsg::setMessageType()
	{
		if(!downLinkNasTransportPdu) downLinkNasTransportPdu = (Ngap_NGAP_PDU_t*)calloc(1, sizeof(Ngap_NGAP_PDU_t));

		MessageType downLinkNasTransportPduTypeIE;
		downLinkNasTransportPduTypeIE.setProcedureCode(Ngap_ProcedureCode_id_DownlinkNASTransport);
		downLinkNasTransportPduTypeIE.setTypeOfMessage(Ngap_NGAP_PDU_PR_initiatingMessage); 
		downLinkNasTransportPduTypeIE.setCriticality(Ngap_Criticality_ignore);
		downLinkNasTransportPduTypeIE.setValuePresent(Ngap_InitiatingMessage__value_PR_DownlinkNASTransport);
		
		if(downLinkNasTransportPduTypeIE.getProcedureCode()==Ngap_ProcedureCode_id_DownlinkNASTransport && downLinkNasTransportPduTypeIE.getTypeOfMessage()==Ngap_NGAP_PDU_PR_initiatingMessage && downLinkNasTransportPduTypeIE.getCriticality()==Ngap_Criticality_ignore)
		{
			downLinkNasTransportPduTypeIE.encode2pdu(downLinkNasTransportPdu);
			downLinkNasTransportIEs = &(downLinkNasTransportPdu->choice.initiatingMessage->value.choice.DownlinkNASTransport);
		}
		else
		{
			cout<<"[warning] This information doesn't refer to DownlinkNASTransport Message!!!"<<endl;
		}
	}
	void DownLinkNasTransportMsg::setAmfUeNgapId(unsigned long id)
	{
		if(!amfUeNgapId)
			amfUeNgapId = new AMF_UE_NGAP_ID();
		amfUeNgapId->setAMF_UE_NGAP_ID(id);

		Ngap_DownlinkNASTransport_IEs_t *ie = (Ngap_DownlinkNASTransport_IEs_t *)calloc(1,sizeof(Ngap_DownlinkNASTransport_IEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_AMF_UE_NGAP_ID;
		ie->criticality = Ngap_Criticality_reject;
		ie->value.present = Ngap_DownlinkNASTransport_IEs__value_PR_AMF_UE_NGAP_ID;
		
		int ret = amfUeNgapId->encode2AMF_UE_NGAP_ID(ie->value.choice.AMF_UE_NGAP_ID);
		if(!ret)
		{
			cout<<"encode AMF_UE_NGAP_ID IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&downLinkNasTransportIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode AMF_UE_NGAP_ID IE error"<<endl;
	}
	void DownLinkNasTransportMsg::setRanUeNgapId(uint32_t ran_ue_ngap_id)
	{
		if(!ranUeNgapId)
			ranUeNgapId = new RAN_UE_NGAP_ID();
		ranUeNgapId->setRanUeNgapId(ran_ue_ngap_id);

		Ngap_DownlinkNASTransport_IEs_t *ie = (Ngap_DownlinkNASTransport_IEs_t *)calloc(1,sizeof(Ngap_DownlinkNASTransport_IEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_RAN_UE_NGAP_ID;
		ie->criticality = Ngap_Criticality_reject;
		ie->value.present = Ngap_DownlinkNASTransport_IEs__value_PR_RAN_UE_NGAP_ID;
		
		int ret = ranUeNgapId->encode2RAN_UE_NGAP_ID(ie->value.choice.RAN_UE_NGAP_ID);
		if(!ret)
		{
			cout<<"encode RAN_UE_NGAP_ID IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&downLinkNasTransportIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode RAN_UE_NGAP_ID IE error"<<endl;
	}
	void DownLinkNasTransportMsg::setOldAmfName(const std::string name)
	{
		if(!oldAmfName)
			oldAmfName = new AmfName();
		oldAmfName->setValue(name);
	
		Ngap_DownlinkNASTransport_IEs_t *ie = (Ngap_DownlinkNASTransport_IEs_t *)calloc(1,sizeof(Ngap_DownlinkNASTransport_IEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_OldAMF;
		ie->criticality = Ngap_Criticality_reject;
		ie->value.present = Ngap_DownlinkNASTransport_IEs__value_PR_AMFName;
		
		int ret = oldAmfName->encode2AmfName(&ie->value.choice.AMFName);
		if(!ret)
		{
			cout<<"encode oldAmfName IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&downLinkNasTransportIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode oldAmfName IE error"<<endl;
	}
	void DownLinkNasTransportMsg::setRanPagingPriority(uint8_t pagingPriority)
	{
		if(!ranPagingPriority)
			ranPagingPriority = new RANPagingPriority();
		ranPagingPriority->setRANPagingPriority(pagingPriority);

		Ngap_DownlinkNASTransport_IEs_t *ie = (Ngap_DownlinkNASTransport_IEs_t *)calloc(1,sizeof(Ngap_DownlinkNASTransport_IEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_RANPagingPriority;
		ie->criticality = Ngap_Criticality_ignore;
		ie->value.present = Ngap_DownlinkNASTransport_IEs__value_PR_RANPagingPriority;
		
		int ret = ranPagingPriority->encode2RANPagingPriority(ie->value.choice.RANPagingPriority);
		if(!ret)
		{
			cout<<"encode RANPagingPriority IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&downLinkNasTransportIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode RANPagingPriority IE error"<<endl;
	}
	void DownLinkNasTransportMsg::setNasPdu(uint8_t *nas,size_t sizeofnas)
	{
		if(!nasPdu)
			nasPdu = new NAS_PDU();

		nasPdu->setNasPdu(nas,sizeofnas);
		
		Ngap_DownlinkNASTransport_IEs_t *ie = (Ngap_DownlinkNASTransport_IEs_t *)calloc(1,sizeof(Ngap_DownlinkNASTransport_IEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_NAS_PDU;
		ie->criticality = Ngap_Criticality_reject;
		ie->value.present = Ngap_DownlinkNASTransport_IEs__value_PR_NAS_PDU;
		
		int ret = nasPdu->encode2octetstring(ie->value.choice.NAS_PDU);
		if(!ret)
		{
			cout<<"encode NAS_PDU IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&downLinkNasTransportIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode NAS_PDU IE error"<<endl;
	}
	void DownLinkNasTransportMsg::setIndex2Rat_Frequency_SelectionPriority(uint8_t value)
	{
		if(!indexToRFSP)
			indexToRFSP = new IndexToRFSP();
		indexToRFSP->setIndexToRFSP(value);

		Ngap_DownlinkNASTransport_IEs_t *ie = (Ngap_DownlinkNASTransport_IEs_t *)calloc(1,sizeof(Ngap_DownlinkNASTransport_IEs_t));
		ie->id = Ngap_ProtocolIE_ID_id_IndexToRFSP;
		ie->criticality = Ngap_Criticality_ignore;
		ie->value.present = Ngap_DownlinkNASTransport_IEs__value_PR_IndexToRFSP;
		
		int ret = indexToRFSP->encode2IndexToRFSP(ie->value.choice.IndexToRFSP);
		if(!ret)
		{
			cout<<"encode IndexToRFSP IE error"<<endl;
			return ;
		}
		
		ret = ASN_SEQUENCE_ADD(&downLinkNasTransportIEs->protocolIEs.list, ie);
    	if( ret != 0) cout<<"encode IndexToRFSP IE error"<<endl;
	}
	
	int  DownLinkNasTransportMsg::encode2buffer(uint8_t *buf, int buf_size)
	{
		asn_fprint(stderr, &asn_DEF_Ngap_NGAP_PDU, downLinkNasTransportPdu);
		asn_enc_rval_t er = aper_encode_to_buffer(&asn_DEF_Ngap_NGAP_PDU, NULL, downLinkNasTransportPdu, buf, buf_size);
    	cout<<"er.encoded("<<er.encoded<<")"<<endl;
		return er.encoded;
	}
	//Decapsulation
	bool DownLinkNasTransportMsg::decodefrompdu(Ngap_NGAP_PDU_t *ngap_msg_pdu)
	{
  		downLinkNasTransportPdu = ngap_msg_pdu;
		
	    if(downLinkNasTransportPdu->present == Ngap_NGAP_PDU_PR_initiatingMessage)
		{
			if(downLinkNasTransportPdu->choice.initiatingMessage && downLinkNasTransportPdu->choice.initiatingMessage->procedureCode == Ngap_ProcedureCode_id_DownlinkNASTransport && downLinkNasTransportPdu->choice.initiatingMessage->criticality == Ngap_Criticality_ignore && downLinkNasTransportPdu->choice.initiatingMessage->value.present == Ngap_InitiatingMessage__value_PR_DownlinkNASTransport)
			{
				downLinkNasTransportIEs = &downLinkNasTransportPdu->choice.initiatingMessage->value.choice.DownlinkNASTransport;
			}
			else
			{
				cout<<"Check DownlinkNASTransport message error!!!"<<endl;
                return false;
			}
		}
		else
		{
			cout<<"MessageType error!!!"<<endl;
            return false;
		}
		for(int i=0; i< downLinkNasTransportIEs->protocolIEs.list.count; i++)
		{
			switch(downLinkNasTransportIEs->protocolIEs.list.array[i]->id)
			{
				case Ngap_ProtocolIE_ID_id_AMF_UE_NGAP_ID:{
					if(downLinkNasTransportIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_reject && downLinkNasTransportIEs->protocolIEs.list.array[i]->value.present == Ngap_DownlinkNASTransport_IEs__value_PR_AMF_UE_NGAP_ID)
					{
						amfUeNgapId = new AMF_UE_NGAP_ID();
						if(!amfUeNgapId->decodefromAMF_UE_NGAP_ID(downLinkNasTransportIEs->protocolIEs.list.array[i]->value.choice.AMF_UE_NGAP_ID))
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
					if(downLinkNasTransportIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_reject && downLinkNasTransportIEs->protocolIEs.list.array[i]->value.present == Ngap_DownlinkNASTransport_IEs__value_PR_RAN_UE_NGAP_ID)
					{
						ranUeNgapId = new RAN_UE_NGAP_ID();
						if(!ranUeNgapId->decodefromRAN_UE_NGAP_ID(downLinkNasTransportIEs->protocolIEs.list.array[i]->value.choice.RAN_UE_NGAP_ID))
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
				case Ngap_ProtocolIE_ID_id_OldAMF:{
					if(downLinkNasTransportIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_reject && downLinkNasTransportIEs->protocolIEs.list.array[i]->value.present == Ngap_DownlinkNASTransport_IEs__value_PR_AMFName)
					{
						oldAmfName = new AmfName();
						if(!oldAmfName->decodefromAmfName(&downLinkNasTransportIEs->protocolIEs.list.array[i]->value.choice.AMFName))
						{
							cout<<"decoded ngap OldAMFName IE error"<<endl;
							return false;
						}
					}
					else
					{
						cout<<"decoded ngap OldAMFName IE error"<<endl;
						return false;
					}
				}break;
				case Ngap_ProtocolIE_ID_id_RANPagingPriority:{
					if(downLinkNasTransportIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_ignore && downLinkNasTransportIEs->protocolIEs.list.array[i]->value.present == Ngap_DownlinkNASTransport_IEs__value_PR_RANPagingPriority)
					{
						ranPagingPriority = new RANPagingPriority();
						if(!ranPagingPriority->decodefromRANPagingPriority(downLinkNasTransportIEs->protocolIEs.list.array[i]->value.choice.RANPagingPriority))
						{
							cout<<"decoded ngap RANPagingPriority IE error"<<endl;
							return false;
						}
					}
					else
					{
						cout<<"decoded ngap RANPagingPriority IE error"<<endl;
						return false;
					}
				}break;
				case Ngap_ProtocolIE_ID_id_NAS_PDU:{
					if(downLinkNasTransportIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_reject && downLinkNasTransportIEs->protocolIEs.list.array[i]->value.present == Ngap_DownlinkNASTransport_IEs__value_PR_NAS_PDU)
					{
						nasPdu = new NAS_PDU();
						if(!nasPdu->decodefromoctetstring(downLinkNasTransportIEs->protocolIEs.list.array[i]->value.choice.NAS_PDU))
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
				case Ngap_ProtocolIE_ID_id_IndexToRFSP:{
					if(downLinkNasTransportIEs->protocolIEs.list.array[i]->criticality == Ngap_Criticality_ignore && downLinkNasTransportIEs->protocolIEs.list.array[i]->value.present == Ngap_DownlinkNASTransport_IEs__value_PR_IndexToRFSP)
					{
						indexToRFSP = new IndexToRFSP();
						if(!indexToRFSP->decodefromIndexToRFSP(downLinkNasTransportIEs->protocolIEs.list.array[i]->value.choice.IndexToRFSP))
						{
							cout<<"decoded ngap IndexToRFSP IE error"<<endl;
							return false;
						}
					}
					else
					{
						cout<<"decoded ngap IndexToRFSP IE error"<<endl;
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
  	}
	unsigned long DownLinkNasTransportMsg::getAmfUeNgapId()
	{
		return amfUeNgapId->getAMF_UE_NGAP_ID();
	}
	uint32_t DownLinkNasTransportMsg::getRanUeNgapId()
	{
		return ranUeNgapId->getRanUeNgapId();
	}
	bool DownLinkNasTransportMsg::getOldAmfName(std::string &name)
	{
		if(!oldAmfName) return false;
		oldAmfName->getValue(name);
		return true;
	}
	uint8_t DownLinkNasTransportMsg::getRanPagingPriority()
	{
		return ranPagingPriority->getRANPagingPriority();
	}
	bool DownLinkNasTransportMsg::getNasPdu(uint8_t *&nas,size_t &sizeofnas)
	{		
		if(!nasPdu->getNasPdu(nas,sizeofnas)) return false;
		
		return true;
	}
	uint8_t DownLinkNasTransportMsg::getIndex2Rat_Frequency_SelectionPriority()
	{
		return indexToRFSP->getIndexToRFSP();
	}
	
}


