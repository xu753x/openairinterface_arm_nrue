#ifndef _UESECURITYCAPABILITIES_H_
#define _UESECURITYCAPABILITIES_H_

extern "C"{
  #include "Ngap_UESecurityCapabilities.h"
}

namespace ngap{

	class UESecurityCapabilities{
	public:
		UESecurityCapabilities();
		virtual ~UESecurityCapabilities();
		
		void setUESecurityCapabilities(uint16_t m_NR_EncryptionAlgs, uint16_t m_NR_IntegrityProtectionAlgs, uint16_t m_E_UTRA_EncryptionAlgs, uint16_t m_E_UTRA_IntegrityProtectionAlgs);
		bool getUESecurityCapabilities(uint16_t &m_NR_EncryptionAlgs, uint16_t &m_NR_IntegrityProtectionAlgs, uint16_t &m_E_UTRA_EncryptionAlgs, uint16_t &m_E_UTRA_IntegrityProtectionAlgs);		
		
		bool encode2UESecurityCapabilities(Ngap_UESecurityCapabilities_t &);
		bool decodefromUESecurityCapabilities(Ngap_UESecurityCapabilities_t &);
	private:
		uint16_t NR_EncryptionAlgs;
		uint16_t NR_IntegrityProtectionAlgs;
		uint16_t E_UTRA_EncryptionAlgs;
		uint16_t E_UTRA_IntegrityProtectionAlgs;
};

}

#endif


