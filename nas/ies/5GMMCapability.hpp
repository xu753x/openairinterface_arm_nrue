#ifndef __5GMMCapability_H_
#define __5GMMCapability_H_

#include <stdint.h>

namespace nas{

class _5GMMCapability{
public:
  _5GMMCapability();
  _5GMMCapability(const uint8_t iei, uint8_t value);
  ~_5GMMCapability();
  void setValue(uint8_t iei, uint8_t value);
  int encode2buffer(uint8_t *buf, int len);
  int decodefrombuffer(uint8_t *buf, int len, bool is_option);
  uint8_t getValue();
private:
  uint8_t m_iei;
  uint8_t m_value;
  int length;

};






}









#endif
