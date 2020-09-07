#ifndef _CONFIDENTIALITYPROTECTIONRESULT_H_
#define _CONFIDENTIALITYPROTECTIONRESULT_H_

extern "C"{
  #include "Ngap_ConfidentialityProtectionResult.h"
}

namespace ngap{

	class ConfidentialityProtectionResult{
	public:
		ConfidentialityProtectionResult();
		virtual ~ConfidentialityProtectionResult();
		
		void setConfidentialityProtectionResult(e_Ngap_ConfidentialityProtectionResult m_confidentialityProtectionResult);
		bool getConfidentialityProtectionResult(long &m_confidentialityProtectionResult);
		
		bool encode2ConfidentialityProtectionResult(Ngap_ConfidentialityProtectionResult_t &);
		bool decodefromConfidentialityProtectionResult(Ngap_ConfidentialityProtectionResult_t);
	private:
		long confidentialityProtectionResult;
	};

}

#endif


