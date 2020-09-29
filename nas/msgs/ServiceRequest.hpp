#ifndef _SERVICE_REQUEST_H_
#define _SERVICE_REQUEST_H_

#include "nas_ie_header.hpp"
#include <string>
#include "bstrlib.h"

using namespace std;

namespace nas{

class ServiceRequest{
public:
  ServiceRequest();
  ~ServiceRequest();
public:
  void setHeader(uint8_t security_header_type);
  void setngKSI(uint8_t tsc, uint8_t key_set_id);
  void setServiceType(uint8_t stp);
  void set5G_S_TMSI(uint16_t amfSetId, uint8_t amfPointer, string tmsi);
  void setUplink_data_status(uint16_t value);
  void setPDU_session_status(uint16_t value);
  void setAllowed_PDU_Session_Status(uint16_t value);
  void setNAS_Message_Container(bstring value);
  int encode2buffer(uint8_t *buf, int len);
public:
  int decodefrombuffer(NasMmPlainHeader * header, uint8_t *buf, int len);
  uint8_t getngKSI();//return -1;表示获取IE失败
  uint8_t getServiceType();
  bool get5G_S_TMSI(uint16_t &amfSetId, uint8_t &amfPointer, string &tmsi);
  uint16_t getUplinkDataStatus();//return -1;表示获取IE失败;
  uint16_t getPduSessionStatus();
  uint16_t getAllowedPduSessionStatus();
  bool getNasMessageContainer(bstring &nas);
private:
  NasMmPlainHeader      *plain_header;
  NasKeySetIdentifier   *ie_ngKSI;
  ServiceType           *ie_service_type;
  _5GSMobilityIdentity  *ie_5g_s_tmsi;
  UplinkDataStatus      *ie_uplink_data_status;
  PDU_Session_Status    *ie_PDU_session_status;
  Allowed_PDU_Session_Status *ie_allowed_PDU_session_status;
  NAS_Message_Container  *ie_nas_message_container;
};


}
















#endif
