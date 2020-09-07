#ifndef _AMF_UE_NGAP_ID_H_
#define _AMF_UE_NGAP_ID_H_

extern "C"{
  #include "Ngap_AMF-UE-NGAP-ID.h"
}

namespace ngap{

	class AMF_UE_NGAP_ID{
	public:
		AMF_UE_NGAP_ID();
		virtual ~AMF_UE_NGAP_ID();
		
		void setAMF_UE_NGAP_ID(unsigned long);
		unsigned long getAMF_UE_NGAP_ID();
		
		bool encode2AMF_UE_NGAP_ID(Ngap_AMF_UE_NGAP_ID_t &);
		bool decodefromAMF_UE_NGAP_ID(Ngap_AMF_UE_NGAP_ID_t &);
	private:
		unsigned long amfUeNgapId;
};

}

#endif

