#ifndef _TAC_H_
#define _TAC_H_

extern "C"{
  #include "Ngap_TAC.h"
}

namespace ngap{

	class TAC{
	public:
		TAC();
		virtual ~TAC();
		bool encode2octetstring(Ngap_TAC_t&);
		bool decodefromoctetstring(Ngap_TAC_t &m_tac);
		uint32_t getTac();
		void setTac(uint32_t m_tac);
	private:
		uint32_t tac:24;
	};

}













#endif
