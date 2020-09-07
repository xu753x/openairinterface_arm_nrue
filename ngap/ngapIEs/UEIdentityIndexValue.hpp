#ifndef _UEIDENTITYINDEXVALUE_H_
#define _UEIDENTITYINDEXVALUE_H_

extern "C"{
  #include "Ngap_UEIdentityIndexValue.h"
}

namespace ngap{

	class UEIdentityIndexValue{
	public:
		UEIdentityIndexValue();
		virtual ~UEIdentityIndexValue();
		
		void setUEIdentityIndexValue(uint16_t m_indexLength10/*10bits限制*/);
		void getUEIdentityIndexValue(uint16_t &m_indexLength10/*10bits限制*/);
		
		bool encode2UEIdentityIndexValue(Ngap_UEIdentityIndexValue_t *ueIdentityIndexValue);
		bool decodefromUEIdentityIndexValue(Ngap_UEIdentityIndexValue_t *ueIdentityIndexValue);
	private:
		uint16_t indexLength10;
};

}

#endif


