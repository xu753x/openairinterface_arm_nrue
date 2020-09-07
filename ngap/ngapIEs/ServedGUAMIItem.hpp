#ifndef _SERVEDGUAMIITEM_H_
#define _SERVEDGUAMIITEM_H_

#include "GUAMI.hpp"
#include "AMFName.hpp"

extern "C"{
  #include "Ngap_ServedGUAMIItem.h"
}

namespace ngap{

	class ServedGUAMIItem{
	public:
		ServedGUAMIItem();
		virtual ~ServedGUAMIItem();
		
		void setGUAMI(GUAMI *);
		void setBackupAMFName(AmfName *);
		
		bool encode2ServedGUAMIItem(Ngap_ServedGUAMIItem *);
		bool decodefromServedGUAMIItem(Ngap_ServedGUAMIItem *);

		void getGUAMI(GUAMI* &);
		bool getBackupAMFName(AmfName* &);
	private:
		GUAMI *guamiGroup;
		AmfName *backupAMFName;
		bool backupAMFNameIsSet;
	};

}



#endif

