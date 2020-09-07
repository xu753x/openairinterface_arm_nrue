#ifndef _NOTIFICATIONCONTROL_H_
#define _NOTIFICATIONCONTROL_H_

extern "C"{
  #include "Ngap_NotificationControl.h"
}

namespace ngap{

	class NotificationControl{
	public:
		NotificationControl();
		virtual ~NotificationControl();
		
		void setNotificationControl(e_Ngap_NotificationControl value);
		bool getNotificationControl(e_Ngap_NotificationControl &value);
		
		bool encode2NotificationControl(Ngap_NotificationControl_t *);
		bool decodefromNotificationControl(Ngap_NotificationControl_t *);
	private:
		long notificationcontrol;
	};

}

#endif


