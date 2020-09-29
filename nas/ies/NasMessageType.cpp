#include "NasMessageType.hpp"


using namespace nas;

void NasMessageType::setValue(const uint8_t type){
  m_type = type;
}

uint8_t NasMessageType::getValue(){
  return m_type;
}
