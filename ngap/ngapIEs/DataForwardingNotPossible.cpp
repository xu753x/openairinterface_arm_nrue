#include "DataForwardingNotPossible.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	DataForwardingNotPossible::DataForwardingNotPossible(){}  

	DataForwardingNotPossible::~DataForwardingNotPossible(){} 
	
	#if 0
	void DataForwardingNotPossible::setDataForwardingNotPossible(const uint32_t m_gtp_teid)
	{
		gtp_teid = m_gtp_teid;
	}
	bool DataForwardingNotPossible::getDataForwardingNotPossible(uint32_t &m_gtp_teid)
	{
		m_gtp_teid = gtp_teid;
		
		return true;
	}
	#endif
	
	bool DataForwardingNotPossible::encode2DataForwardingNotPossible(Ngap_DataForwardingNotPossible_t &dataForwardingNotPossible)
	{
		dataForwardingNotPossible = Ngap_DataForwardingNotPossible_data_forwarding_not_possible;

		return true;
	}
	bool DataForwardingNotPossible::decodefromDataForwardingNotPossible(Ngap_DataForwardingNotPossible_t &dataForwardingNotPossible)
	{
		if(dataForwardingNotPossible != Ngap_DataForwardingNotPossible_data_forwarding_not_possible) return false;
		
		return true;
	}

}




