#ifndef _SupportedTaItem_H
#define _SupportedTaItem_H

#include "BroadcastPLMNItem.hpp"
#include "Tac.hpp"

extern "C"{
  #include "Ngap_SupportedTAItem.h"
}

namespace ngap{

	class SupportedTaItem{
	public:
		SupportedTaItem();
		virtual ~SupportedTaItem();
		
		void setTac(TAC *m_tac);
		void getTac(TAC *&m_tac);
		void setBroadcastPlmnList(BroadcastPLMNItem * m_broadcastPLMNItem, int numOfItem);
		void getBroadcastPlmnList(BroadcastPLMNItem * & m_broadcastPLMNItem, int & numOfItem);
		bool encode2SupportedTaItem(Ngap_SupportedTAItem_t *ta);
		bool decodefromSupportedTaItem(Ngap_SupportedTAItem_t *ta);
	private:
		TAC *tac;
		BroadcastPLMNItem * broadcastPLMNItem;
		int numberOfBroadcastItem;
	};

}



#endif
