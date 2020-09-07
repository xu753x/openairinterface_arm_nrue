#ifndef _DEFAULT_PAGING_DRX_H_
#define _DEFAULT_PAGING_DRX_H_

extern "C"{
  #include "Ngap_PagingDRX.h"
}

namespace ngap{

	class DefaultPagingDRX{
	public:
		DefaultPagingDRX();
		virtual ~DefaultPagingDRX();
		
		void setValue(e_Ngap_PagingDRX m_pagingDrx);
		int  getValue();
		bool encode2DefaultPagingDRX(Ngap_PagingDRX_t &);
		bool decodefromDefaultPagingDRX(Ngap_PagingDRX_t &);
	private:
		int pagingDrx;
	};

}







#endif
