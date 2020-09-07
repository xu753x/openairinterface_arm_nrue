#ifndef _NETWORKINSTANCE_H_
#define _NETWORKINSTANCE_H_

extern "C"{
  #include "Ngap_NetworkInstance.h"
}

namespace ngap{

	class NetworkInstance{
	public:
		NetworkInstance();
		virtual ~NetworkInstance();
		
		void setNetworkInstance(long m_networkinstance);
		bool getNetworkInstance(long &m_networkinstance);
		
		bool encode2NetworkInstance(Ngap_NetworkInstance_t &);
		bool decodefromNetworkInstance(Ngap_NetworkInstance_t );
	private:
		long networkinstance;
};

}

#endif


