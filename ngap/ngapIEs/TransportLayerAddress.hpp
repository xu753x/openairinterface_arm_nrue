#ifndef _TRANSPORTLAYERADDRESS_H_
#define _TRANSPORTLAYERADDRESS_H_

#include <string>

extern "C"{
  #include "Ngap_TransportLayerAddress.h"
}

namespace ngap{

	class TransportLayerAddress{
	public:
		TransportLayerAddress();
		virtual ~TransportLayerAddress();
		
		void setTransportLayerAddress(const std::string m_ipaddress);
		bool getTransportLayerAddress(std::string &m_ipaddress);		
		
		bool encode2TransportLayerAddress(Ngap_TransportLayerAddress_t &transportLayerAddress);
		bool decodefromTransportLayerAddress(Ngap_TransportLayerAddress_t &transportLayerAddress);
	private:
		std::string ipaddress;
};

}

#endif


