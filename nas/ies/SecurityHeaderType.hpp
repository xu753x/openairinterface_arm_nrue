#ifndef _SECURITY_HEADER_TYPE_H_
#define _SECURITY_HEADER_TYPE_H_

#include <stdint.h>

namespace nas{

class SecurityHeaderType{
public:
  void setValue(const uint8_t value);
  uint8_t getValue();
private:
  uint8_t secu_header_type:4;
};







}









#endif
