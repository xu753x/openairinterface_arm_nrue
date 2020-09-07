#ifndef _IESCRITICALITYDIAGNOSTICS_H_
#define _IESCRITICALITYDIAGNOSTICS_H_

extern "C"{
  #include "Ngap_CriticalityDiagnostics-IE-Item.h"
}

namespace ngap{

	class IEsCriticalityDiagnostics{
	public:
		IEsCriticalityDiagnostics();
		virtual ~IEsCriticalityDiagnostics();
		
		void encode2pdu(Ngap_CriticalityDiagnostics_IE_Item_t *);
		void decodefrompdu(Ngap_CriticalityDiagnostics_IE_Item_t *);
		void setIeCriticality(Ngap_Criticality_t);
		void setIeId(Ngap_ProtocolIE_ID_t);
		void setTypeOfError(Ngap_TypeOfError_t);
		Ngap_Criticality_t getIeCriticality();
		Ngap_ProtocolIE_ID_t getIeId();
		Ngap_TypeOfError_t getTypeOfError();
	private:
		Ngap_Criticality_t criticality;
		Ngap_ProtocolIE_ID_t protocolIE_ID;
		Ngap_TypeOfError_t typeOfError;
	};
}

#endif
