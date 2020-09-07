#ifndef _DATAFORWARDINGNOTPOSSIBLE_H_
#define _DATAFORWARDINGNOTPOSSIBLE_H_

extern "C"{
  #include "Ngap_DataForwardingNotPossible.h"
}

namespace ngap{

	class DataForwardingNotPossible{
	public:
		DataForwardingNotPossible();
		virtual ~DataForwardingNotPossible();
		
		//void setDataForwardingNotPossible(const bool m_gtp_teid);
		//bool getDataForwardingNotPossible(uint32_t &m_gtp_teid);		
		
		bool encode2DataForwardingNotPossible(Ngap_DataForwardingNotPossible_t &dataForwardingNotPossible);
		bool decodefromDataForwardingNotPossible(Ngap_DataForwardingNotPossible_t &dataForwardingNotPossible);
	};

}

#endif

