#ifndef __5GSTrackingAreaIdList_H_
#define __5GSTrackingAreaIdList_H_

#include <vector>
#include "struct.hpp"

namespace nas{

class _5GSTrackingAreaIdList{
public:
  _5GSTrackingAreaIdList(uint8_t iei, std::vector<p_tai_t> tai_list);
  int encode2buffer(uint8_t *buf, int len);
private:
  uint8_t m_iei;
  std::vector<p_tai_t> m_tai_list;
private:
  int encode_00_type(p_tai_t item, uint8_t *buf, int len);
  int encode_01_type(p_tai_t item, uint8_t *buf, int len);
  int encode_10_type(p_tai_t item, uint8_t *buf, int len);
  int encode_mcc_mnc(nas_plmn_t plmn, uint8_t *buf, int len);
};


}












#endif
