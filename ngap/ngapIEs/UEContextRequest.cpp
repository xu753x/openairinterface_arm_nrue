#include "UEContextRequest.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	UEContextRequest::UEContextRequest()
	{
		uecontextrequest = -1;
	}  

	UEContextRequest::~UEContextRequest(){} 

	void UEContextRequest::setUEContextRequest(e_Ngap_UEContextRequest m_uecontextrequest)
	{
		uecontextrequest = m_uecontextrequest;
	}
	int UEContextRequest::getUEContextRequest()
	{
                cout<<"testing 11.2"<<endl;
		return uecontextrequest;
	}
	
	bool UEContextRequest::encode2UEContextRequest(Ngap_UEContextRequest_t &uEContextRequest)
	{
		uEContextRequest = uecontextrequest;
		return true;
	}
	bool UEContextRequest::decodefromUEContextRequest(Ngap_UEContextRequest_t &uEContextRequest)
	{
		uecontextrequest = uEContextRequest;
		return true;
	}
}




