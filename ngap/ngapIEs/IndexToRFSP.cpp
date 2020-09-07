#include "IndexToRFSP.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	IndexToRFSP::IndexToRFSP(){}  

	IndexToRFSP::~IndexToRFSP(){} 

	void IndexToRFSP::setIndexToRFSP(uint8_t m_indexToRFSP)
	{
		indexToRFSP = m_indexToRFSP;
	}
	uint8_t IndexToRFSP::getIndexToRFSP()
	{
		return indexToRFSP;
	}
	
	bool IndexToRFSP::encode2IndexToRFSP(Ngap_IndexToRFSP_t &indextorfsp)
	{
		indextorfsp = indexToRFSP;
		return true;
	}
	bool IndexToRFSP::decodefromIndexToRFSP(Ngap_IndexToRFSP_t &indextorfsp)
	{
		indexToRFSP = indextorfsp;
		return true;
	}
}




