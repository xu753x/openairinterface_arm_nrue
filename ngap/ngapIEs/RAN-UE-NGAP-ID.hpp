#ifndef _RAN_UE_NGAP_ID_H_
#define _RAN_UE_NGAP_ID_H_

extern "C"{
  #include "Ngap_RAN-UE-NGAP-ID.h"
}

namespace ngap{

	class RAN_UE_NGAP_ID{
	public:
		RAN_UE_NGAP_ID();
		virtual ~RAN_UE_NGAP_ID();
		
		void setRanUeNgapId(uint32_t);
		uint32_t getRanUeNgapId();
		
		bool encode2RAN_UE_NGAP_ID(Ngap_RAN_UE_NGAP_ID_t &);
		bool decodefromRAN_UE_NGAP_ID(Ngap_RAN_UE_NGAP_ID_t &);
	private:
		uint32_t ranUeNgapId;
};

}

#endif

