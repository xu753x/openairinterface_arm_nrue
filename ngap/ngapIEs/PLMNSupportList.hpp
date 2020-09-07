#ifndef _PLMNSUPPORTLIST_H_
#define _PLMNSUPPORTLIST_H_

#include "PLMNSupportItem.hpp"

extern "C"{
	#include "Ngap_PLMNSupportList.h"
}

namespace ngap{

	class PLMNSupportList{
	public:
		PLMNSupportList();
		virtual ~PLMNSupportList();
		
		bool encode2PLMNSupportList(Ngap_PLMNSupportList_t *);
		bool decodefromPLMNSupportList(Ngap_PLMNSupportList_t *);
		void addPLMNSupportItems(PLMNSupportItem * m_plmnsupportItemItem, int numOfItem);
		void getPLMNSupportItems(PLMNSupportItem * &m_plmnsupportItemItem, int &numOfItem);
	private:
		PLMNSupportItem * plmnsupportItemItem;
		int numberOfplmnsupportItemItem;
	};

}

#endif
