#ifndef _BroadcastPLMNItem_H
#define _BroadcastPLMNItem_H

#include "PlmnId.hpp"
#include "S-NSSAI.hpp"

extern "C"{
  #include "Ngap_BroadcastPLMNItem.h"
}

namespace ngap{

	class BroadcastPLMNItem{
	public:
		BroadcastPLMNItem();
		virtual ~BroadcastPLMNItem();
		
		void setPlmnSliceSupportList(PlmnId * m_plmn, S_NSSAI * snssai, int num);
		void getPlmnSliceSupportList(PlmnId * & m_plmn, S_NSSAI * & m_snssai, int & snssainum);
		bool encode2BroadcastPLMNItem(Ngap_BroadcastPLMNItem_t *);
		bool decodefromBroadcastPLMNItem( Ngap_BroadcastPLMNItem_t * pdu);
	private:
		PlmnId *plmn;
		S_NSSAI * snssai;
		int numOfSnssai;
	};
}
#endif
