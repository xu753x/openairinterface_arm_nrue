#ifndef _PLMNSUPPORTITEM_H_
#define _PLMNSUPPORTITEM_H_

#include "PlmnId.hpp"
#include "S-NSSAI.hpp"

extern "C"{
  #include "Ngap_PLMNSupportItem.h"
}

namespace ngap{

	class PLMNSupportItem{
	public:
		PLMNSupportItem();
		virtual ~PLMNSupportItem();
		
		void setPlmnSliceSupportList(PlmnId * m_plmn, S_NSSAI * m_snssai, int num);
		
		bool encode2PLMNSupportItem(Ngap_PLMNSupportItem_t *);
		bool decodefromPLMNSupportItem(Ngap_PLMNSupportItem_t *);

		void getPlmnSliceSupportList(PlmnId * & m_plmn, S_NSSAI * & m_snssai, int & snssainum);
	private:
		PlmnId *plmn;
		S_NSSAI * snssai;
		int numOfSnssai;
};

}



#endif

