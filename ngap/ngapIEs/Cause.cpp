#include "Cause.hpp"

#include<iostream>
using namespace std;

namespace ngap{

	Cause::Cause()
	{
		causeValue = -1;
	}
	Cause::~Cause(){}


	void Cause::setChoiceOfCause(Ngap_Cause_PR m_causePresent)
	{
		causePresent = m_causePresent;
	}

	void Cause::setValue(long m_causeValue)
	{
		causeValue = m_causeValue;
	}

	bool Cause::encode2Cause(Ngap_Cause_t *cause)
	{
		cause->present = causePresent;
		switch(causePresent)
		{
			case Ngap_Cause_PR_radioNetwork:{
				cause->choice.radioNetwork = causeValue;
			break;}
			case Ngap_Cause_PR_transport:{
				cause->choice.transport = causeValue;
			break;}
			case Ngap_Cause_PR_nas:{
				cause->choice.nas = causeValue;
			break;}
			case Ngap_Cause_PR_protocol:{
				cause->choice.protocol = causeValue;
			break;}
			case Ngap_Cause_PR_misc:{
				cause->choice.misc = causeValue;
			break;}
			default:{
				cout<<"[Warning] Cause Present error!"<<endl;
				return false;
			break;}
		}
		return true;		
	}
	
	bool Cause::decodefromCause(Ngap_Cause_t * pdu)
	{
		causePresent = pdu->present;
		switch(causePresent)
		{
			case Ngap_Cause_PR_radioNetwork:{
				causeValue = pdu->choice.radioNetwork;
			}break;
			case Ngap_Cause_PR_transport:{
				causeValue = pdu->choice.transport;
			}break;
			case Ngap_Cause_PR_nas:{
				causeValue = pdu->choice.nas;
			}break;
			case Ngap_Cause_PR_protocol:{
				causeValue = pdu->choice.protocol;
			}break;
			case Ngap_Cause_PR_misc:{
				causeValue = pdu->choice.misc;
			}break;
			default:{
				cout<<"[Warning] Cause Present error!"<<endl;
				return false;
			}
		}
		return true;
	}
	Ngap_Cause_PR Cause::getChoiceOfCause()
	{
		return causePresent;
	}
	long Cause::getValue()
	{
		return causeValue;
	}
}
