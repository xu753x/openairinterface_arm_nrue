#ifndef _NONDYNAMIC5QIDESCRIPTOR_H_
#define _NONDYNAMIC5QIDESCRIPTOR_H_

#include "FiveQI.hpp"
#include "PriorityLevelQos.hpp"
#include "AveragingWindow.hpp"
#include "MaximumDataBurstVolume.hpp"

extern "C"{
  #include "Ngap_NonDynamic5QIDescriptor.h"
}

namespace ngap{

	class NonDynamic5QIDescriptor{
	public:
		NonDynamic5QIDescriptor();
		virtual ~NonDynamic5QIDescriptor();
		
		void setNonDynamic5QIDescriptor(FiveQI *m_fiveQI,PriorityLevelQos *m_priorityLevelQos,AveragingWindow *m_averagingWindow,MaximumDataBurstVolume	*m_maximumDataBurstVolume);
		bool getNonDynamic5QIDescriptor(FiveQI *&m_fiveQI,PriorityLevelQos *&m_priorityLevelQos,AveragingWindow *&m_averagingWindow,MaximumDataBurstVolume	*&m_maximumDataBurstVolume);
		
		bool encode2NonDynamic5QIDescriptor(Ngap_NonDynamic5QIDescriptor_t *);
		bool decodefromNonDynamic5QIDescriptor(Ngap_NonDynamic5QIDescriptor_t *);
	private:
		FiveQI *fiveQI;
		PriorityLevelQos *priorityLevelQos;	/* OPTIONAL */
		AveragingWindow *averagingWindow;	/* OPTIONAL */
		MaximumDataBurstVolume *maximumDataBurstVolume;	/* OPTIONAL */
	};
}

#endif
