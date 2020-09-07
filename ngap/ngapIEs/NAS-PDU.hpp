#ifndef _NAS_PDU_H_
#define _NAS_PDU_H_

extern "C"{
  #include "Ngap_NAS-PDU.h"
}

namespace ngap{

	class NAS_PDU{
	public:
		NAS_PDU();
		virtual ~NAS_PDU();

		bool encode2octetstring(Ngap_NAS_PDU_t&);
		bool decodefromoctetstring(Ngap_NAS_PDU_t &);
		bool getNasPdu(uint8_t* &buffer,size_t &size);
		void setNasPdu(uint8_t* buffer,size_t size);
	private:
		char *naspdubuffer;
		size_t buffersize;
	};

}

#endif
