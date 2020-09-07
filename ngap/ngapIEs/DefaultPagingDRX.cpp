#include "DefaultPagingDRX.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	DefaultPagingDRX::DefaultPagingDRX(){}
	DefaultPagingDRX::~DefaultPagingDRX(){}
	
	void DefaultPagingDRX::setValue(e_Ngap_PagingDRX   m_pagingDrx)
	{
		pagingDrx = m_pagingDrx;
	}
	bool DefaultPagingDRX::encode2DefaultPagingDRX(Ngap_PagingDRX_t & pagingdrx)
	{
		pagingdrx = pagingDrx;
		return true;
	}

	bool DefaultPagingDRX::decodefromDefaultPagingDRX(Ngap_PagingDRX_t &pagingdrx)
	{
		pagingDrx = pagingdrx;
		return true; 
	}
	int DefaultPagingDRX::getValue()
	{
		return pagingDrx;
	}

}
