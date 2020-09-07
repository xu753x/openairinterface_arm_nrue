#ifndef _ASSOCIATEDQOSFLOWLIST_H_
#define _ASSOCIATEDQOSFLOWLIST_H_

#include "AssociatedQosFlowItem.hpp"

extern "C"{
  #include "Ngap_AssociatedQosFlowList.h"
}

namespace ngap{

	class AssociatedQosFlowList{
	public:
		AssociatedQosFlowList();
		virtual ~AssociatedQosFlowList();
		
		void setAssociatedQosFlowList(AssociatedQosFlowItem *m_associatedQosFlowItem,int numofitem);
		bool getAssociatedQosFlowList(AssociatedQosFlowItem *&m_associatedQosFlowItem,int &numofitem);		
		
		bool encode2AssociatedQosFlowList(Ngap_AssociatedQosFlowList_t &associatedQosFlowList);
		bool decodefromAssociatedQosFlowList(Ngap_AssociatedQosFlowList_t &associatedQosFlowList);
	private:
		AssociatedQosFlowItem *associatedQosFlowItem;
		int numofassociatedQosFlowItem;
	};

}

#endif
