#ifndef _SERVICE_ACCEPT_H_
#define _SERVICE_ACCEPT_H_

#include <stdint.h>
#include "nas_ie_header.hpp"
#include<string>

namespace nas{

class ServiceAccept{
public:
  ServiceAccept();
  ~ServiceAccept();
public:
  void setHeader(uint8_t security_header_type);
  void setPDU_session_status(uint16_t value);
  void setPDU_session_reactivation_result(uint16_t);
  int encode2buffer(uint8_t *buf, int len);
private:
  NasMmPlainHeader      *plain_header;
  PDU_Session_Status    *ie_PDU_session_status;
  PDU_Session_Reactivation_Result * ie_session_reactivation_result;
};


}
















#endif
