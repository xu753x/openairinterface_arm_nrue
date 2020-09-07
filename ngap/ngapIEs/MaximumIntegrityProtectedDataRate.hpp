#ifndef _MAXIMUMINTEGRITYPROTECTEDDATARATE_H_
#define _MAXIMUMINTEGRITYPROTECTEDDATARATE_H_

extern "C"{
  #include "Ngap_MaximumIntegrityProtectedDataRate.h"
}

namespace ngap{

	class MaximumIntegrityProtectedDataRate{
	public:
		MaximumIntegrityProtectedDataRate();
		virtual ~MaximumIntegrityProtectedDataRate();
		
		void setMaximumIntegrityProtectedDataRate(e_Ngap_MaximumIntegrityProtectedDataRate m_maxintprotdatarate);
		bool getMaximumIntegrityProtectedDataRate(long &m_maxintprotdatarate);
		
		bool encode2MaximumIntegrityProtectedDataRate(Ngap_MaximumIntegrityProtectedDataRate_t &);
		bool decodefromMaximumIntegrityProtectedDataRate(Ngap_MaximumIntegrityProtectedDataRate_t );
	private:
		long maxintprotdatarate;
	};

}

#endif


