#include "nas_mm_plain_header.hpp"
#include "logger.hpp"
#include "3gpp_ts24501.hpp"

using namespace nas;

void NasMmPlainHeader::setEpdIE(const uint8_t epd){
  ie_epd.setValue(epd);
}

uint8_t NasMmPlainHeader::getEpdIE(){
  return ie_epd.getValue();
}

void NasMmPlainHeader::setSecurityHeaderTypeIE(const uint8_t type){
  ie_secu_header_type.setValue(type);
}

uint8_t NasMmPlainHeader::getSecurityHeaderTypeIE(){
  return ie_secu_header_type.getValue();
}

void NasMmPlainHeader::setMessageTypeIE(const uint8_t type){
  ie_msg_type.setValue(type);
}

uint8_t NasMmPlainHeader::getMessageType(){
  return msg_type;
}

void NasMmPlainHeader::setHeader(uint8_t epd_,uint8_t security_header_type, uint8_t msg_type_){
  epd = epd_;
  secu_header_type = security_header_type;
  msg_type = msg_type_;
}

int NasMmPlainHeader::encode2buffer(uint8_t *buf, int len){
  Logger::nas_mm().debug("encoding NasMmPlainHeader");
  if(len < 3){
    Logger::nas_mm().error("[encoding nas mm header error][buffer length is less than 3 octets]");
    return 0;
  }else{
    *(buf++) = epd;
    *(buf++) = secu_header_type;
    *(buf++) = msg_type;
    Logger::nas_mm().debug("encoded NasMmPlainHeader len((3 octets))");
    return 3;
  }
}

int NasMmPlainHeader::decodefrombuffer(uint8_t *buf, int len){
  Logger::nas_mm().debug("decoding NasMmPlainHeader");
  if(len < 3){
    Logger::nas_mm().error("[decoding nas mm header error][buffer length is less than 3 octets]");
    return 0;
  }else{
    epd = *(buf++);
    secu_header_type = *(buf++);
    msg_type = *(buf++);
  }
  Logger::nas_mm().debug("decoded NasMmPlainHeader len(3 octets)");
  return 3;
}
