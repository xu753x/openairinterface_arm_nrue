#ifndef _QOSFLOWIDENTIFIER_H_
#define _QOSFLOWIDENTIFIER_H_

extern "C"{
  #include "Ngap_QosFlowIdentifier.h"
}

namespace ngap{

	class QosFlowIdentifier{
	public:
		QosFlowIdentifier();
		virtual ~QosFlowIdentifier();
		
		void setQosFlowIdentifier(long value);
		bool getQosFlowIdentifier(long &value);
		
		bool encode2QosFlowIdentifier(Ngap_QosFlowIdentifier_t *);
		bool decodefromQosFlowIdentifier(Ngap_QosFlowIdentifier_t *);
	private:
		long identifier;
	};

}

#endif

