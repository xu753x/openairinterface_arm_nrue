#include "SecurityHeaderType.hpp"


using namespace nas;

void SecurityHeaderType::setValue(const uint8_t value){
  secu_header_type = 0x0f & value;
}
uint8_t SecurityHeaderType::getValue(){
  return secu_header_type & 0x0f;
}
