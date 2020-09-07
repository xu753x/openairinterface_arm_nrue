#ifndef _SUPPORTED_TA_LIST_H_
#define _SUPPORTED_TA_LIST_H_

extern "C"{
  #include "Ngap_SupportedTAList.h"
}

#include "SupportedTaItem.hpp"

namespace ngap{

	class SupportedTAList{
	public:
		SupportedTAList();
		virtual ~SupportedTAList();
		
		bool encode2SupportedTAList(Ngap_SupportedTAList_t * ngSetupRequest);
		bool decodefromSupportedTAList(Ngap_SupportedTAList_t *pdu);
		void setSupportedTaItems(SupportedTaItem * m_supportedTaItem, int numOfItem);
		void getSupportedTaItems(SupportedTaItem * &m_supportedTaItem, int &numOfItem);
	private:
		SupportedTaItem * supportedTaItem;
		int numberOfSupportedTaItem;
	};
}

#endif
