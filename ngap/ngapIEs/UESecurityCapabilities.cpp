#include "UESecurityCapabilities.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	UESecurityCapabilities::UESecurityCapabilities(){}  

	UESecurityCapabilities::~UESecurityCapabilities(){} 
	
	
	void UESecurityCapabilities::setUESecurityCapabilities(uint16_t m_NR_EncryptionAlgs, uint16_t m_NR_IntegrityProtectionAlgs, uint16_t m_E_UTRA_EncryptionAlgs, uint16_t m_E_UTRA_IntegrityProtectionAlgs)
	{
		NR_EncryptionAlgs =  m_NR_EncryptionAlgs;
		NR_IntegrityProtectionAlgs = m_NR_IntegrityProtectionAlgs;
		E_UTRA_EncryptionAlgs = m_E_UTRA_EncryptionAlgs;
		E_UTRA_IntegrityProtectionAlgs = m_E_UTRA_IntegrityProtectionAlgs;
	}
	bool UESecurityCapabilities::getUESecurityCapabilities(uint16_t &m_NR_EncryptionAlgs, uint16_t &m_NR_IntegrityProtectionAlgs, uint16_t &m_E_UTRA_EncryptionAlgs, uint16_t &m_E_UTRA_IntegrityProtectionAlgs)
	{
		m_NR_EncryptionAlgs =  NR_EncryptionAlgs;
		m_NR_IntegrityProtectionAlgs = NR_IntegrityProtectionAlgs;
		m_E_UTRA_EncryptionAlgs = E_UTRA_EncryptionAlgs;
		m_E_UTRA_IntegrityProtectionAlgs = E_UTRA_IntegrityProtectionAlgs;
		
		return true;
	}
	
	bool UESecurityCapabilities::encode2UESecurityCapabilities(Ngap_UESecurityCapabilities_t &ueSecurityCapabilities)
	{
		ueSecurityCapabilities.nRencryptionAlgorithms.bits_unused = 0;
		ueSecurityCapabilities.nRencryptionAlgorithms.size = sizeof(uint16_t);
		ueSecurityCapabilities.nRencryptionAlgorithms.buf = (uint8_t *)calloc(1,ueSecurityCapabilities.nRencryptionAlgorithms.size);
		if(!ueSecurityCapabilities.nRencryptionAlgorithms.buf ) return false;
		for(int i=0;i<ueSecurityCapabilities.nRencryptionAlgorithms.size;i++)
		{
			ueSecurityCapabilities.nRencryptionAlgorithms.buf[i] = (NR_EncryptionAlgs & (0xff00>>i*8))>>((ueSecurityCapabilities.nRencryptionAlgorithms.size-i-1)*8);
		} 
	
		ueSecurityCapabilities.nRintegrityProtectionAlgorithms.bits_unused = 0;
		ueSecurityCapabilities.nRintegrityProtectionAlgorithms.size = sizeof(uint16_t);
		ueSecurityCapabilities.nRintegrityProtectionAlgorithms.buf = (uint8_t *)calloc(1,ueSecurityCapabilities.nRintegrityProtectionAlgorithms.size);
		if(!ueSecurityCapabilities.nRintegrityProtectionAlgorithms.buf ) return false;
		for(int i=0;i<ueSecurityCapabilities.nRintegrityProtectionAlgorithms.size;i++)
		{
			ueSecurityCapabilities.nRintegrityProtectionAlgorithms.buf[i] = (NR_IntegrityProtectionAlgs & (0xff00>>i*8))>>((ueSecurityCapabilities.nRintegrityProtectionAlgorithms.size-i-1)*8);
		} 
		
		ueSecurityCapabilities.eUTRAencryptionAlgorithms.bits_unused = 0;
		ueSecurityCapabilities.eUTRAencryptionAlgorithms.size = sizeof(uint16_t);
		ueSecurityCapabilities.eUTRAencryptionAlgorithms.buf = (uint8_t *)calloc(1,ueSecurityCapabilities.eUTRAencryptionAlgorithms.size);
		if(!ueSecurityCapabilities.eUTRAencryptionAlgorithms.buf ) return false;
		for(int i=0;i<ueSecurityCapabilities.eUTRAencryptionAlgorithms.size;i++)
		{
			ueSecurityCapabilities.eUTRAencryptionAlgorithms.buf[i] = (E_UTRA_EncryptionAlgs & (0xff00>>i*8))>>((ueSecurityCapabilities.eUTRAencryptionAlgorithms.size-i-1)*8);
		} 

		ueSecurityCapabilities.eUTRAintegrityProtectionAlgorithms.bits_unused = 0;
		ueSecurityCapabilities.eUTRAintegrityProtectionAlgorithms.size = sizeof(uint16_t);
		ueSecurityCapabilities.eUTRAintegrityProtectionAlgorithms.buf = (uint8_t *)calloc(1,ueSecurityCapabilities.eUTRAintegrityProtectionAlgorithms.size);
		if(!ueSecurityCapabilities.eUTRAintegrityProtectionAlgorithms.buf ) return false;
		for(int i=0;i<ueSecurityCapabilities.eUTRAintegrityProtectionAlgorithms.size;i++)
		{
			ueSecurityCapabilities.eUTRAintegrityProtectionAlgorithms.buf[i] = (E_UTRA_IntegrityProtectionAlgs & (0xff00>>i*8))>>((ueSecurityCapabilities.eUTRAintegrityProtectionAlgorithms.size-i-1)*8);
		} 

		return true;
	}
	bool UESecurityCapabilities::decodefromUESecurityCapabilities(Ngap_UESecurityCapabilities_t &ueSecurityCapabilities)
	{
		if(!ueSecurityCapabilities.nRencryptionAlgorithms.buf) return false;
		if(!ueSecurityCapabilities.nRintegrityProtectionAlgorithms.buf) return false;
		if(!ueSecurityCapabilities.eUTRAencryptionAlgorithms.buf) return false;
		if(!ueSecurityCapabilities.eUTRAintegrityProtectionAlgorithms.buf) return false;

		NR_EncryptionAlgs = 0;
		NR_IntegrityProtectionAlgs = 0;
		E_UTRA_EncryptionAlgs = 0;
		E_UTRA_IntegrityProtectionAlgs = 0;
		
		for(int i=0;i<ueSecurityCapabilities.nRencryptionAlgorithms.size;i++)
		{
			NR_EncryptionAlgs = NR_EncryptionAlgs<<8;
			NR_EncryptionAlgs |= ueSecurityCapabilities.nRencryptionAlgorithms.buf[i];
		}
		for(int i=0;i<ueSecurityCapabilities.nRintegrityProtectionAlgorithms.size;i++)
		{
			NR_IntegrityProtectionAlgs = NR_IntegrityProtectionAlgs<<8;
			NR_IntegrityProtectionAlgs |= ueSecurityCapabilities.nRintegrityProtectionAlgorithms.buf[i];
		}
		for(int i=0;i<ueSecurityCapabilities.eUTRAencryptionAlgorithms.size;i++)
		{
			E_UTRA_EncryptionAlgs = E_UTRA_EncryptionAlgs<<8;
			E_UTRA_EncryptionAlgs |= ueSecurityCapabilities.eUTRAencryptionAlgorithms.buf[i];
		}
		for(int i=0;i<ueSecurityCapabilities.eUTRAintegrityProtectionAlgorithms.size;i++)
		{
			E_UTRA_IntegrityProtectionAlgs = E_UTRA_IntegrityProtectionAlgs<<8;
			E_UTRA_IntegrityProtectionAlgs |= ueSecurityCapabilities.eUTRAintegrityProtectionAlgorithms.buf[i];
		}

		return true;
	}

}
