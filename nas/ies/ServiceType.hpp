#ifndef _SERVICE_TYPE_H_
#define _SERVICE_TYPE_H_

#include <stdint.h>

namespace nas{

class ServiceType{
public:
  ServiceType();
  ~ServiceType();
  ServiceType(uint8_t iei, uint8_t stp);
public:
  int encode2buffer(uint8_t *buf, int len);
  int decodefrombuffer(uint8_t *nuf, int len, bool is_optional, bool is_high);
  uint8_t getValue();
private:
  uint8_t _iei;
  uint8_t value;
};

}

















#endif
