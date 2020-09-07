#include "UPTransportLayerInformation.hpp"

extern "C"{
  #include "Ngap_GTPTunnel.h"
}

#include <iostream>
using namespace std;

namespace ngap{

	UpTransportLayerInformation::UpTransportLayerInformation()
	{
		transportLayerAddress = NULL;
		gtpTeid = NULL;
	}  

	UpTransportLayerInformation::~UpTransportLayerInformation(){} 
	
	
	void UpTransportLayerInformation::setUpTransportLayerInformation(TransportLayerAddress *m_transportLayerAddress,GtpTeid *m_gtpTeid)
	{
		transportLayerAddress = m_transportLayerAddress;
		gtpTeid = m_gtpTeid;
	}
	bool UpTransportLayerInformation::getUpTransportLayerInformation(TransportLayerAddress *&m_transportLayerAddress,GtpTeid *&m_gtpTeid)
	{
		m_transportLayerAddress = transportLayerAddress;
		m_gtpTeid = gtpTeid;
		
		return true;
	}
	
	bool UpTransportLayerInformation::encode2UpTransportLayerInformation(Ngap_UPTransportLayerInformation_t &upTransportLayerInfo)
	{
		upTransportLayerInfo.present = Ngap_UPTransportLayerInformation_PR_gTPTunnel;
		Ngap_GTPTunnel_t *gtptunnel = (Ngap_GTPTunnel_t *)calloc(1,sizeof(Ngap_GTPTunnel_t));
		if(!gtptunnel) return false;
		if(!transportLayerAddress->encode2TransportLayerAddress(gtptunnel->transportLayerAddress)) return false;
		if(!gtpTeid->encode2GtpTeid(gtptunnel->gTP_TEID)) return false;
		upTransportLayerInfo.choice.gTPTunnel = gtptunnel;

		return true;
	}
	bool UpTransportLayerInformation::decodefromUpTransportLayerInformation(Ngap_UPTransportLayerInformation_t &upTransportLayerInfo)
	{
		if(upTransportLayerInfo.present != Ngap_UPTransportLayerInformation_PR_gTPTunnel) return false;
		if(!upTransportLayerInfo.choice.gTPTunnel) return false;
		transportLayerAddress = new TransportLayerAddress();
		gtpTeid = new GtpTeid();
		
		if(!transportLayerAddress->decodefromTransportLayerAddress(upTransportLayerInfo.choice.gTPTunnel->transportLayerAddress)) return false;
		if(!gtpTeid->decodefromGtpTeid(upTransportLayerInfo.choice.gTPTunnel->gTP_TEID)) return false;

		return true;
	}

}

