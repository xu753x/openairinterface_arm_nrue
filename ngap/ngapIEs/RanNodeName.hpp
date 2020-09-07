#ifndef _RANNODENAME_H_
#define _RANNODENAME_H_

extern "C"{
  #include "Ngap_RANNodeName.h"
}

#include <string>

namespace ngap{

	class RanNodeName{
	public:
		RanNodeName();
		virtual ~RanNodeName();
		
		void setValue(const std::string ranName);
		bool encode2RanNodeName(Ngap_RANNodeName_t *);
		bool decodefromRanNodeName(Ngap_RANNodeName_t *);
		bool getValue(std::string &ranName);
	private:
		char *ranNodeName;
	};

}

#endif
