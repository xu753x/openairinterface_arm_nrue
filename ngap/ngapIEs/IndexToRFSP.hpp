#ifndef _INDEXTORFSP_H_
#define _INDEXTORFSP_H_

extern "C"{
  #include "Ngap_IndexToRFSP.h"
}

namespace ngap{

	class IndexToRFSP{
	public:
		IndexToRFSP();
		virtual ~IndexToRFSP();
		
		void setIndexToRFSP(uint8_t);
		uint8_t getIndexToRFSP();
		
		bool encode2IndexToRFSP(Ngap_IndexToRFSP_t &);
		bool decodefromIndexToRFSP(Ngap_IndexToRFSP_t &);
	private:
		uint8_t indexToRFSP;
	};

}

#endif
