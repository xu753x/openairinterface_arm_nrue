#include "MICOModeIndication.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	MICOModeIndication::MICOModeIndication(){}
	MICOModeIndication::~MICOModeIndication(){}
	
	bool MICOModeIndication::encode2MICOModeIndication(Ngap_MICOModeIndication_t *micoModeIndication)
	{
		if(!micoModeIndication) return false;
		*micoModeIndication = Ngap_MICOModeIndication_true;
		
		return true;
	}

	bool MICOModeIndication::decodefromMICOModeIndication(Ngap_MICOModeIndication_t *micoModeIndication)
	{
		if(!micoModeIndication) return false;
		if(*micoModeIndication != Ngap_MICOModeIndication_true) return false;
		
		return true;
	}

}

