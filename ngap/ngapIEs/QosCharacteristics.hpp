#ifndef _QOSCHARACTERISTICS_H_
#define _QOSCHARACTERISTICS_H_

#include "NonDynamic5QIDescriptor.hpp"
#include "Dynamic5QIDescriptor.hpp"

extern "C"{
  #include "Ngap_QosCharacteristics.h"
}

namespace ngap{

	class QosCharacteristics{
	public:
		QosCharacteristics();
		virtual ~QosCharacteristics();
		
		void setQosCharacteristics(NonDynamic5QIDescriptor *m_nonDynamic5QIDescriptor);
		void setQosCharacteristics(Dynamic5QIDescriptor *m_dynamic5QIDescriptor);
		int  QosCharacteristicsPresent();
		bool getQosCharacteristics(NonDynamic5QIDescriptor *&m_nonDynamic5QIDescriptor);
		bool getQosCharacteristics(Dynamic5QIDescriptor *&m_dynamic5QIDescriptor);
		bool encode2QosCharacteristics(Ngap_QosCharacteristics_t *);
		bool decodefromQosCharacteristics(Ngap_QosCharacteristics_t *);
	private:
		NonDynamic5QIDescriptor *nonDynamic5QIDescriptor;
		Dynamic5QIDescriptor *dynamic5QIDescriptor;
	};
}

#endif


