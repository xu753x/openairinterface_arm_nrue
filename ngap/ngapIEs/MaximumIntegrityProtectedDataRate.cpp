#include "MaximumIntegrityProtectedDataRate.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	MaximumIntegrityProtectedDataRate::MaximumIntegrityProtectedDataRate(){}  

	MaximumIntegrityProtectedDataRate::~MaximumIntegrityProtectedDataRate(){} 
	
	
	void MaximumIntegrityProtectedDataRate::setMaximumIntegrityProtectedDataRate(e_Ngap_MaximumIntegrityProtectedDataRate m_maxintprotdatarate)
	{
		maxintprotdatarate = m_maxintprotdatarate;
	}
	bool MaximumIntegrityProtectedDataRate::getMaximumIntegrityProtectedDataRate(long &m_maxintprotdatarate)
	{
		m_maxintprotdatarate = maxintprotdatarate;
		
		return true;
	}
	
	bool MaximumIntegrityProtectedDataRate::encode2MaximumIntegrityProtectedDataRate(Ngap_MaximumIntegrityProtectedDataRate_t &maximumIntegrityProtectedDataRate)
	{
		maximumIntegrityProtectedDataRate = maxintprotdatarate;

		return true;
	}
	bool MaximumIntegrityProtectedDataRate::decodefromMaximumIntegrityProtectedDataRate(Ngap_MaximumIntegrityProtectedDataRate_t maximumIntegrityProtectedDataRate)
	{
		maxintprotdatarate = maximumIntegrityProtectedDataRate;
		
		return true;
	}

}
