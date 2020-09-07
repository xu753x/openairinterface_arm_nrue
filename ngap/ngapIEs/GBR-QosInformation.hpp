#ifndef _GBR_QOSINFORMATION_H_
#define _GBR_QOSINFORMATION_H_

#include "NotificationControl.hpp"
#include "PacketLossRate.hpp"

extern "C"{
  #include "Ngap_GBR-QosInformation.h"
}

namespace ngap{

	class GBR_QosInformation{
	public:
		GBR_QosInformation();
		virtual ~GBR_QosInformation();
		
		void setGBR_QosInformation(long m_maximumFlowBitRateDL,long m_maximumFlowBitRateUL,long m_guaranteedFlowBitRateDL,long m_guaranteedFlowBitRateUL,
											 NotificationControl *m_notificationControl,PacketLossRate *m_maximumPacketLossRateDL,PacketLossRate *m_maximumPacketLossRateUL);
		bool getGBR_QosInformation(long &m_maximumFlowBitRateDL,long &m_maximumFlowBitRateUL,long &m_guaranteedFlowBitRateDL,long &m_guaranteedFlowBitRateUL,
											 NotificationControl *&m_notificationControl,PacketLossRate *&m_maximumPacketLossRateDL,PacketLossRate *&m_maximumPacketLossRateUL);
		
		bool encode2GBR_QosInformation(Ngap_GBR_QosInformation_t *);
		bool decodefromGBR_QosInformation(Ngap_GBR_QosInformation_t *);
	private:
		long maximumFlowBitRateDL;
		long maximumFlowBitRateUL;
		long guaranteedFlowBitRateDL;
		long guaranteedFlowBitRateUL;
		NotificationControl *notificationControl; /* OPTIONAL */
		PacketLossRate *maximumPacketLossRateDL; /* OPTIONAL */
		PacketLossRate *maximumPacketLossRateUL; /* OPTIONAL */
	};
}

#endif
