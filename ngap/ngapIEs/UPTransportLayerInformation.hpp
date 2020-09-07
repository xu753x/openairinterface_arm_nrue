#ifndef _UPTRANSPORTLAYERINFORMATION_H_
#define _UPTRANSPORTLAYERINFORMATION_H_

#include "TransportLayerAddress.hpp"
#include "GTP-TEID.hpp"

extern "C"{
  #include "Ngap_UPTransportLayerInformation.h"
}

namespace ngap{

	class UpTransportLayerInformation{
	public:
		UpTransportLayerInformation();
		virtual ~UpTransportLayerInformation();
		
		void setUpTransportLayerInformation(TransportLayerAddress *m_transportLayerAddress,GtpTeid *m_gtpTeid);
		bool getUpTransportLayerInformation(TransportLayerAddress *&m_transportLayerAddress,GtpTeid *&m_gtpTeid);		
		
		bool encode2UpTransportLayerInformation(Ngap_UPTransportLayerInformation_t &upTransportLayerInfo);
		bool decodefromUpTransportLayerInformation(Ngap_UPTransportLayerInformation_t &upTransportLayerInfo);
	private:
		TransportLayerAddress *transportLayerAddress;
		GtpTeid *gtpTeid;
};

}

#endif

