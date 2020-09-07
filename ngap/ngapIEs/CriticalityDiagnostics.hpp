#ifndef _CRITICALITYDIAGNOSTICS_H_
#define _CRITICALITYDIAGNOSTICS_H_

#include "IEsCriticalityDiagnostics.hpp"

extern "C"{
  #include "Ngap_NGSetupFailure.h"
  #include "Ngap_ProtocolIE-Field.h"
  #include "Ngap_CriticalityDiagnostics.h"
  #include "Ngap_CriticalityDiagnostics-IE-List.h"
}

namespace ngap{

	class CriticalityDiagnostics{
	public:
		CriticalityDiagnostics();
		virtual ~CriticalityDiagnostics();
		
		int encode2pdu(Ngap_NGSetupFailure_t *);
		bool decodefrompdu(Ngap_CriticalityDiagnostics_t *);
		void setProcedureCodeValue(Ngap_ProcedureCode_t);
		void setTriggeringMessageValue(Ngap_TriggeringMessage_t);
		void setCriticalityValue(Ngap_Criticality_t);
		void setIEsCriticalityDiagnosticsList(IEsCriticalityDiagnostics *m_iEsCriticalityDiagnostics,int num);
		bool getProcedureCodeValue(Ngap_ProcedureCode_t&);
		bool getTriggeringMessageValue(Ngap_TriggeringMessage_t&);
		bool getCriticalityValue(Ngap_Criticality_t&);
		bool getIEsCriticalityDiagnosticsList(IEsCriticalityDiagnostics* &m_iEsCriticalityDiagnostics,int &num);
	private:
		Ngap_ProcedureCode_t procedureCode;	/* OPTIONAL */
		Ngap_TriggeringMessage_t triggeringMessage;	/* OPTIONAL */
		Ngap_Criticality_t procedureCriticality;	/* OPTIONAL */
		//Ngap_CriticalityDiagnostics_IE_List_t iEsCriticalityDiagnostics;	/* OPTIONAL */
		IEsCriticalityDiagnostics *iEsCriticalityDiagnostics;

		bool procedureCodeIsSet;
		bool triggeringMessageIsSet;
		bool procedureCriticalityIsSet;
		int numberOfIEsCriticalityDiagnostics;
	};
}

#endif