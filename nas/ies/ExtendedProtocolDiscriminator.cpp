#include "ExtendedProtocolDiscriminator.hpp"

using namespace nas;

void ExtendedProtocolDiscriminator::setValue(const uint8_t epd){
  m_epd = epd;
}
uint8_t ExtendedProtocolDiscriminator::getValue(){
  return m_epd;
}

void ExtendedProtocolDiscriminator::encode2buffer(uint8_t *buf, int len){
  
}
