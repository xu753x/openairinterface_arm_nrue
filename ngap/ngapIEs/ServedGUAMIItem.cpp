#include "ServedGUAMIItem.hpp"

#include <iostream>
using namespace std;

namespace ngap{
	ServedGUAMIItem::ServedGUAMIItem()
	{
		guamiGroup = NULL;
		backupAMFName = NULL;
		backupAMFNameIsSet = false;
	}
	ServedGUAMIItem::~ServedGUAMIItem(){}

	void ServedGUAMIItem::setGUAMI(GUAMI *m_guami)
	{
		guamiGroup = m_guami;
	}
	void ServedGUAMIItem::setBackupAMFName(AmfName *m_backupAMFName)
	{
		backupAMFNameIsSet = true;
		backupAMFName = m_backupAMFName;
	}
	
	bool ServedGUAMIItem::encode2ServedGUAMIItem(Ngap_ServedGUAMIItem *servedGUAMIItem)
	{
		if(!guamiGroup->encode2GUAMI(&(servedGUAMIItem->gUAMI))) return false;
		if(backupAMFNameIsSet)
		{	
			Ngap_AMFName_t	*backupamfname = (Ngap_AMFName_t *)calloc(1,sizeof(Ngap_AMFName_t));
			if(!backupamfname) return false;
			if(!backupAMFName->encode2AmfName(backupamfname)) return false;
			servedGUAMIItem->backupAMFName = backupamfname;
			
		}
		return true;
	}
	bool ServedGUAMIItem::decodefromServedGUAMIItem(Ngap_ServedGUAMIItem *pdu)
	{
		guamiGroup = new GUAMI();
		if(!guamiGroup->decodefromGUAMI(&pdu->gUAMI)) return false;
		if(pdu->backupAMFName)
		{
			backupAMFNameIsSet = true;
			backupAMFName = new AmfName();
			if(!backupAMFName->decodefromAmfName(pdu->backupAMFName)) return false;
		}
		return true;
	}
	void ServedGUAMIItem::getGUAMI(GUAMI* &m_guami)
	{
		m_guami = guamiGroup;
	}
	bool ServedGUAMIItem::getBackupAMFName(AmfName* &m_backupAMFName)
	{
		m_backupAMFName = backupAMFName;
		return backupAMFNameIsSet;
	}
}
