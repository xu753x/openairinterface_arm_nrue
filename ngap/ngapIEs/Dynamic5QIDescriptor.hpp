#ifndef _DYNAMIC5QIDESCRIPTOR_H_
#define _DYNAMIC5QIDESCRIPTOR_H_

#include "PriorityLevelQos.hpp"
#include "PacketDelayBudget.hpp"
#include "PacketErrorRate.hpp"

#include "FiveQI.hpp"
#include "DelayCritical.hpp"
#include "AveragingWindow.hpp"
#include "MaximumDataBurstVolume.hpp"

extern "C"{
  #include "Ngap_Dynamic5QIDescriptor.h"
}

namespace ngap{

	class Dynamic5QIDescriptor{
	public:
		Dynamic5QIDescriptor();
		virtual ~Dynamic5QIDescriptor();
		
		void setDynamic5QIDescriptor(PriorityLevelQos *m_priorityLevelQos,PacketDelayBudget *m_packetDelayBudget,PacketErrorRate *m_packetErrorRate,
											 	FiveQI *m_fiveQI,DelayCritical *m_delayCritical,AveragingWindow *m_averagingWindow,MaximumDataBurstVolume	*m_maximumDataBurstVolume);
		bool getDynamic5QIDescriptor(PriorityLevelQos *&m_priorityLevelQos,PacketDelayBudget *&m_packetDelayBudget,PacketErrorRate *&m_packetErrorRate,
											 	FiveQI *&m_fiveQI,DelayCritical *&m_delayCritical,AveragingWindow *&m_averagingWindow,MaximumDataBurstVolume	*&m_maximumDataBurstVolume);
		
		bool encode2Dynamic5QIDescriptor(Ngap_Dynamic5QIDescriptor_t *);
		bool decodefromDynamic5QIDescriptor(Ngap_Dynamic5QIDescriptor_t *);
	private:
		PriorityLevelQos *priorityLevelQos;
		PacketDelayBudget *packetDelayBudget;
		PacketErrorRate *packetErrorRate;
		
		FiveQI *fiveQI; /* OPTIONAL */
		DelayCritical *delayCritical;	/* OPTIONAL */
		AveragingWindow *averagingWindow;	/* OPTIONAL */
		MaximumDataBurstVolume *maximumDataBurstVolume;	/* OPTIONAL */

	};
}

#endif

