#ifndef _AMFREGIONID_H_
#define _AMFREGIONID_H_

#include <string>

extern "C"{
  #include "Ngap_AMFRegionID.h"
}

namespace ngap{

	class AMFRegionID{
	public:
		AMFRegionID();
		virtual ~AMFRegionID();
		
		void setAMFRegionID(const std::string );
		void getAMFRegionID(std::string &);
		
		bool encode2bitstring(Ngap_AMFRegionID_t &);
		bool decodefrombitstring(Ngap_AMFRegionID_t &);
	private:
		uint8_t regionid;
};

}

#endif
