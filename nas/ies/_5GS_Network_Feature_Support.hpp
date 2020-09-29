#ifndef __5GS_Network_Feature_Support_H
#define __5GS_Network_Feature_Support_H

#include <stdint.h>

namespace nas{

class _5GS_Network_Feature_Support {
public:
	_5GS_Network_Feature_Support(uint8_t iei);
	_5GS_Network_Feature_Support();
  ~_5GS_Network_Feature_Support();
  _5GS_Network_Feature_Support(const uint8_t iei, uint8_t value, uint8_t value2);
  void setValue(uint8_t value);
  uint8_t getValue();
  int encode2buffer(uint8_t *buf, int len);
  int decodefrombuffer(uint8_t *buf, int len, bool is_option);
private:
  uint8_t _iei;
  uint8_t length;
  uint8_t _value;
  uint8_t _value2;
};







}















#endif


