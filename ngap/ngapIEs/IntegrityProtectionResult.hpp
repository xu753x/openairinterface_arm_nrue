#ifndef _INTEGRITYPROTECTIONRESULT_H_
#define _INTEGRITYPROTECTIONRESULT_H_

extern "C"{
  #include "Ngap_IntegrityProtectionResult.h"
}

namespace ngap{

	class IntegrityProtectionResult{
	public:
		IntegrityProtectionResult();
		virtual ~IntegrityProtectionResult();
		
		void setIntegrityProtectionResult(e_Ngap_IntegrityProtectionResult m_integrityProtectionResult);
		bool getIntegrityProtectionResult(long &m_integrityProtectionResult);
		
		bool encode2IntegrityProtectionResult(Ngap_IntegrityProtectionResult_t &);
		bool decodefromIntegrityProtectionResult(Ngap_IntegrityProtectionResult_t);
	private:
		long integrityProtectionResult;
	};

}

#endif

