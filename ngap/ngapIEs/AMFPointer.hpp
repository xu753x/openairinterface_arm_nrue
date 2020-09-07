#ifndef _AMFPOINTER_H_
#define _AMFPOINTER_H_

#include <string>

extern "C"{
  #include "Ngap_AMFPointer.h"
}

namespace ngap{

	class AMFPointer{
	public:
		AMFPointer();
		virtual ~AMFPointer();
		
		void setAMFPointer(const std::string);
		void getAMFPointer(std::string &);
		
		bool encode2bitstring(Ngap_AMFPointer_t &);
		bool decodefrombitstring(Ngap_AMFPointer_t &);
	private:
		uint8_t pointer;
};

}

#endif
