#ifndef _GTP_TEID_H_
#define _GTP_TEID_H_

extern "C"{
  #include "Ngap_GTP-TEID.h"
}

namespace ngap{

	class GtpTeid{
	public:
		GtpTeid();
		virtual ~GtpTeid();
		
		void setGtpTeid(const uint32_t m_gtp_teid);
		bool getGtpTeid(uint32_t &m_gtp_teid);		
		
		bool encode2GtpTeid(Ngap_GTP_TEID_t &gtpTeid);
		bool decodefromGtpTeid(Ngap_GTP_TEID_t &gtpTeid);
	private:
		uint32_t gtp_teid;
	};

}

#endif
