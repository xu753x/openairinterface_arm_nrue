#ifndef _NAS_MM_PLAIN_HEADER_H_
#define _NAS_MM_PLAIN_HEADER_H_

#include "ExtendedProtocolDiscriminator.hpp"
#include "SecurityHeaderType.hpp"
#include "NasMessageType.hpp"

namespace nas{

class NasMmPlainHeader{
public:
  void setHeader(uint8_t epd_,uint8_t security_header_type, uint8_t msg_type_);
  int encode2buffer(uint8_t *buf, int len);
  int decodefrombuffer(uint8_t *buf, int len);
  void setEpdIE(const uint8_t epd);
  void setSecurityHeaderTypeIE(const uint8_t type);
  void setMessageTypeIE(const uint8_t type);
  uint8_t getEpdIE();
  uint8_t getSecurityHeaderTypeIE();
  uint8_t getMessageType();
private:
  ExtendedProtocolDiscriminator ie_epd;
  SecurityHeaderType            ie_secu_header_type;
  NasMessageType                   ie_msg_type;
  uint8_t        epd;
  uint8_t        secu_header_type;
  uint8_t        msg_type;
};






}










#endif
