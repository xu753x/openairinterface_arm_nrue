#ifndef _MICOMODEINDICATION_H_
#define _MICOMODEINDICATION_H_

extern "C"{
  #include "Ngap_MICOModeIndication.h"
}

namespace ngap{

	class MICOModeIndication{
	public:
		MICOModeIndication();
		virtual ~MICOModeIndication();
		
		//void setMICOModeIndication();
		//bool getMICOModeIndication();
		
		bool encode2MICOModeIndication(Ngap_MICOModeIndication_t *micoModeIndication);
		bool decodefromMICOModeIndication(Ngap_MICOModeIndication_t *micoModeIndication);
	//private:
	//	uint16_t indexLength10;
};

}

#endif



