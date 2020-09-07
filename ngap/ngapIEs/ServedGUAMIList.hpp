#ifndef _SERVEDGUAMILIST_H_
#define _SERVEDGUAMILIST_H_

#include "ServedGUAMIItem.hpp"

extern "C"{
	#include "Ngap_ServedGUAMIList.h"
}

namespace ngap{

	class ServedGUAMIList{
	public:
		ServedGUAMIList();
		virtual ~ServedGUAMIList();
		
		bool encode2ServedGUAMIList(Ngap_ServedGUAMIList_t *);
		bool decodefromServedGUAMIList(Ngap_ServedGUAMIList_t *);
		void addServedGUAMIItems(ServedGUAMIItem * m_servedGUAMIItem, int numOfItem);
		void getServedGUAMIItems(ServedGUAMIItem * &m_servedGUAMIItem, int &numOfItem);
	private:
		ServedGUAMIItem * servedGUAMIItem;
		int numberOfservedGUAMIItem;
	};

}


#endif

