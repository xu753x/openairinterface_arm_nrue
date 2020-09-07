#include "NotificationControl.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	NotificationControl::NotificationControl(){}  

	NotificationControl::~NotificationControl(){} 

	void NotificationControl::setNotificationControl(e_Ngap_NotificationControl value)
	{
		notificationcontrol = value;
	}
	bool NotificationControl::getNotificationControl(e_Ngap_NotificationControl &value)
	{
		if(notificationcontrol == Ngap_NotificationControl_notification_requested)
			value = Ngap_NotificationControl_notification_requested;
		else
			return 0;
		
		return true;
	}
	
	bool NotificationControl::encode2NotificationControl(Ngap_NotificationControl_t *notificationControl)
	{
		*notificationControl = notificationcontrol;
	
		return true;
	}
	bool NotificationControl::decodefromNotificationControl(Ngap_NotificationControl_t *notificationControl)
	{
		notificationcontrol = *notificationControl;
		
		return true;
	}
}
