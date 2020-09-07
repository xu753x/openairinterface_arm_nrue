#ifndef _PDUSESSIONID_H_
#define _PDUSESSIONID_H_

extern "C"{
  #include "Ngap_PDUSessionID.h"
}

namespace ngap{

	class PDUSessionID{
	public:
		PDUSessionID();
		virtual ~PDUSessionID();
		
		void setPDUSessionID(uint8_t m_pdusessionid);
		bool getPDUSessionID(uint8_t &m_pdusessionid);
		
		bool encode2PDUSessionID(Ngap_PDUSessionID_t &pduSessionId);
		bool decodefromPDUSessionID(Ngap_PDUSessionID_t pduSessionId);
	private:
		uint8_t pdusessionid;
};

}

#endif
