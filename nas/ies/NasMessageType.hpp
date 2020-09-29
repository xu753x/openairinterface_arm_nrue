#ifndef _MESSAGE_TYPE_H_
#define _MESSAGE_TYPE_H_

#include<stdint.h>

namespace nas{

class NasMessageType{
public:
  void setValue(const uint8_t type);
  uint8_t getValue();
private:
  uint8_t  m_type;
};





}











#endif
