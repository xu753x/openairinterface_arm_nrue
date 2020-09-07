#ifndef _GLOBALRANNODEID_H_
#define _GLOBALRANNODEID_H_

extern "C"{
  #include "Ngap_GlobalRANNodeID.h"
}

#include "GlobalgNBId.hpp"

namespace ngap{

	class GlobalRanNodeId{
	public:
		GlobalRanNodeId();
		virtual ~GlobalRanNodeId();

		
		void setChoiceOfRanNodeId(Ngap_GlobalRANNodeID_PR m_idPresent);
		void setGlobalgNBID(GlobalgNBId *);
		bool encode2GlobalRANNodeID(Ngap_GlobalRANNodeID_t *);
		bool decodefromGlobalRANNodeID(Ngap_GlobalRANNodeID_t *);
		Ngap_GlobalRANNodeID_PR getChoiceOfRanNodeId();
		void getGlobalgNBID(GlobalgNBId *&);

	private:
		GlobalgNBId *globalgNBId;
		Ngap_GlobalRANNodeID_PR idPresent; 
};


}


#endif
