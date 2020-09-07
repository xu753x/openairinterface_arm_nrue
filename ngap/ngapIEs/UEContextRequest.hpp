#ifndef _UECONTEXTREQUEST_H_
#define _UECONTEXTREQUEST_H_

extern "C"{
  #include "Ngap_UEContextRequest.h"
}

namespace ngap{

	class UEContextRequest{
	public:
		UEContextRequest();
		virtual ~UEContextRequest();
		
		void setUEContextRequest(e_Ngap_UEContextRequest m_uecontextrequest);
		int getUEContextRequest();
		
		bool encode2UEContextRequest(Ngap_UEContextRequest_t &uEContextRequest);
		bool decodefromUEContextRequest(Ngap_UEContextRequest_t &uEContextRequest);
	private:
		int uecontextrequest;
};

}

#endif




