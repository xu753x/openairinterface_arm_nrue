#include "QosCharacteristics.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	QosCharacteristics::QosCharacteristics()
	{
		nonDynamic5QIDescriptor = NULL;
		dynamic5QIDescriptor = NULL;
	}  

	QosCharacteristics::~QosCharacteristics(){} 

	void QosCharacteristics::setQosCharacteristics(NonDynamic5QIDescriptor *m_nonDynamic5QIDescriptor)
	{
		if(nonDynamic5QIDescriptor)
		{
			cout<<"nonDynamic5QIDescriptor has been set!"<<endl;
			return;
		}
		nonDynamic5QIDescriptor = m_nonDynamic5QIDescriptor;
	}
	void QosCharacteristics::setQosCharacteristics(Dynamic5QIDescriptor *m_dynamic5QIDescriptor)
	{
		if(dynamic5QIDescriptor)
		{
			cout<<"Dynamic5QIDescriptor has been set!"<<endl;
			return;
		}
		dynamic5QIDescriptor = m_dynamic5QIDescriptor;
	}
	
	int QosCharacteristics::QosCharacteristicsPresent()
	{
		if(nonDynamic5QIDescriptor)
			return Ngap_QosCharacteristics_PR_nonDynamic5QI;
		else if(dynamic5QIDescriptor)
			return Ngap_QosCharacteristics_PR_dynamic5QI;
		else
			return -1;
	}
	bool QosCharacteristics::getQosCharacteristics(NonDynamic5QIDescriptor *&m_nonDynamic5QIDescriptor)
	{
		if(!nonDynamic5QIDescriptor) return false;
		m_nonDynamic5QIDescriptor = nonDynamic5QIDescriptor;
		
		return true;
	}
	bool QosCharacteristics::getQosCharacteristics(Dynamic5QIDescriptor *&m_dynamic5QIDescriptor)
	{
		if(!dynamic5QIDescriptor) return false;
		m_dynamic5QIDescriptor = dynamic5QIDescriptor;
	
		return true;
	}
	bool QosCharacteristics::encode2QosCharacteristics(Ngap_QosCharacteristics_t *qosCharacteristics)
	{
		if(nonDynamic5QIDescriptor)
		{
			qosCharacteristics->present = Ngap_QosCharacteristics_PR_nonDynamic5QI;
			Ngap_NonDynamic5QIDescriptor_t *nondynamic = (Ngap_NonDynamic5QIDescriptor_t *)calloc(1,sizeof(Ngap_NonDynamic5QIDescriptor_t));
			if(!nondynamic) return false;
			if(!nonDynamic5QIDescriptor->encode2NonDynamic5QIDescriptor(nondynamic)) return false;
			qosCharacteristics->choice.nonDynamic5QI = nondynamic;
		}
		else if(dynamic5QIDescriptor)
		{
			qosCharacteristics->present = Ngap_QosCharacteristics_PR_dynamic5QI;
			Ngap_Dynamic5QIDescriptor_t *dynamic = (Ngap_Dynamic5QIDescriptor_t *)calloc(1,sizeof(Ngap_Dynamic5QIDescriptor_t));
			if(!dynamic) return false;
			if(!dynamic5QIDescriptor->encode2Dynamic5QIDescriptor(dynamic)) return false;
			qosCharacteristics->choice.dynamic5QI = dynamic;
		}
		else
		{
			return false;
		}
		
		return true;
	}
	bool QosCharacteristics::decodefromQosCharacteristics(Ngap_QosCharacteristics_t *qosCharacteristics)
	{
		if(qosCharacteristics->present == Ngap_QosCharacteristics_PR_nonDynamic5QI)
		{
			nonDynamic5QIDescriptor = new NonDynamic5QIDescriptor();
			if(!nonDynamic5QIDescriptor->decodefromNonDynamic5QIDescriptor(qosCharacteristics->choice.nonDynamic5QI)) return false;
		}
		else if(qosCharacteristics->present == Ngap_QosCharacteristics_PR_dynamic5QI)
		{
			dynamic5QIDescriptor = new Dynamic5QIDescriptor();
			if(!dynamic5QIDescriptor->decodefromDynamic5QIDescriptor(qosCharacteristics->choice.dynamic5QI)) return false;
		}
		else
		{
			return false;
		}
		
		return true;
	}
}





