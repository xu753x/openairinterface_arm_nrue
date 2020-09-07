#ifndef _AMFSETID_H_
#define _AMFSETID_H_

#include <string>

extern "C"{
  #include "Ngap_AMFSetID.h"
}

namespace ngap{

	class AMFSetID{
	public:
		AMFSetID();
		virtual ~AMFSetID();
		
		void setAMFSetID(const std::string);
		void getAMFSetID(std::string &);
		
		bool encode2bitstring(Ngap_AMFSetID_t &);
		bool decodefrombitstring(Ngap_AMFSetID_t &);
	private:
		uint16_t setid;
};

}

#endif
