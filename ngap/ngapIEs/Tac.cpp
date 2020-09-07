#include "Tac.hpp"
#include <iostream>
namespace ngap{
	TAC::TAC(){}
	TAC::~TAC(){}

	
	void TAC::setTac(uint32_t m_tac)
	{
		tac = m_tac;
	}
	bool TAC::encode2octetstring(Ngap_TAC_t &m_tac)
	{
		m_tac.size = 3;//OCTET_STRING(SIZE(3))
		//uint8_t *buffer = (uint8_t*)calloc(1,sizeof(uint8_t)+sizeof(uint16_t));
		//if(!buffer) return false;
		
		//*(uint32_t *)buffer = tac & 0x00ffffff;
                //printf("tac(0x%x)\n", tac);
		m_tac.buf = (uint8_t*)calloc(1,sizeof(uint8_t)+sizeof(uint16_t));
		m_tac.buf[2] = tac & 0x0000ff;
		m_tac.buf[1] = (tac & 0x00ff00)>>8;
		m_tac.buf[0] = (tac & 0xff0000)>>16;

		return true;
	}
	bool TAC::decodefromoctetstring(Ngap_TAC_t &m_tac)
	{
		if(!m_tac.buf) return false;
                tac = 0;
                for(int i=0; i< m_tac.size; i++){
                  tac |= m_tac.buf[i] << ((m_tac.size - 1 - i)*8);
                }
                std::cout << "received tac: "<<tac<<std::endl;
		return true;
	}

	uint32_t TAC::getTac()
	{
		return tac;
	}


}
