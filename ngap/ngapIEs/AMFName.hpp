#ifndef _AMFNAME_H_
#define _AMFNAME_H_

#include <string>

extern "C"{
  #include "Ngap_AMFName.h"
}

namespace ngap{

	class AmfName{
	public:
		AmfName();
		virtual ~AmfName();
		
		bool encode2AmfName(Ngap_AMFName_t *);
		bool decodefromAmfName(Ngap_AMFName_t *);
		void setValue(const std::string m_amfName);
  		void getValue(std::string &);
	private:
		char* amfname;
	};
}

#endif

