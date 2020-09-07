#ifndef _GUAMI_H_
#define _GUAMI_H_

#include "PlmnId.hpp"
#include "AMFRegionID.hpp"
#include "AMFSetID.hpp"
#include "AMFPointer.hpp"

extern "C"{
  #include "Ngap_GUAMI.h"
}

namespace ngap{

	class GUAMI{
	public:
		GUAMI();
		virtual ~GUAMI();
		
		void setGUAMI(PlmnId* m_plmnId,AMFRegionID* m_aMFRegionID,AMFSetID* m_aMFSetID,AMFPointer* m_aMFPointer);
		void getGUAMI(PlmnId* &m_plmnId,AMFRegionID* &m_aMFRegionID,AMFSetID* &m_aMFSetID,AMFPointer* &m_aMFPointer);
		
		bool encode2GUAMI(Ngap_GUAMI_t *guami);
		bool decodefromGUAMI(Ngap_GUAMI_t *pdu);
	private:
		PlmnId *plmnId;
		AMFRegionID *aMFRegionID;
		AMFSetID *aMFSetID;
		AMFPointer *aMFPointer;
};

}

#endif
