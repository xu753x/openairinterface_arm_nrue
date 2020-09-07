#ifndef _GlobalgNBId_H
#define _GlobalgNBId_H

#include "PlmnId.hpp"
#include "GNB-ID.hpp"

extern "C"{
  #include "Ngap_GlobalGNB-ID.h"
}

namespace ngap{

	class GlobalgNBId{
	public:
		GlobalgNBId();
		virtual ~GlobalgNBId(); 

		bool encode2GlobalgNBId(Ngap_GlobalGNB_ID_t *);
		bool decodefromGlobalgNBId(Ngap_GlobalGNB_ID_t *);
		void setGlobalgNBId(PlmnId *plmn, GNB_ID *gnbid);
		void getGlobalgNBId(PlmnId *&plmn, GNB_ID *&gnbid);
	private:
		PlmnId *plmnId;
		GNB_ID *gNB_ID;
	};

}

#endif
