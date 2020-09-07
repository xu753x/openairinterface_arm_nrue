#ifndef _PDUSESSIONTYPE_H_
#define _PDUSESSIONTYPE_H_

extern "C"{
  #include "Ngap_PDUSessionType.h"
}

namespace ngap{

	class PDUSessionType{
	public:
		PDUSessionType();
		virtual ~PDUSessionType();
		
		void setPDUSessionType(e_Ngap_PDUSessionType m_pdusessiontype);
		bool getPDUSessionType(long &m_pdusessiontype);
		
		bool encode2PDUSessionType(Ngap_PDUSessionType_t &);
		bool decodefromPDUSessionType(Ngap_PDUSessionType_t );
	private:
		long pdusessiontype;
};

}

#endif

