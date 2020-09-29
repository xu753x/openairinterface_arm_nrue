#ifndef _STRUCT_H_
#define _STRUCT_H_
#include <iostream>
#include <stdint.h>
#include <vector>
extern "C" {
#include "bstrlib.h"
#include "TLVEncoder.h"
#include "TLVDecoder.h"
}
using namespace std;
namespace nas{

typedef struct SNSSAI_s {
         uint8_t sst;
          int32_t sd;//若sd不存在，则sd设置为-1
	int32_t mHplmnSst;//若sd不存在，则sd设置为-1
	int32_t mHplmnSd;//若sd不存在，则sd设置为-1
 }SNSSAI_t;
typedef struct {
	uint8_t ie_type;
	uint8_t ie_len;
	bstring ie_value;
}IE_t;
typedef struct {
	uint8_t ie_type;
	uint16_t ie_len;
	bstring ie_value;
}IE_t_E;
typedef struct {
	uint8_t payloadContainerType : 4;
	std::vector<IE_t> optionalIE;
}PayloadContainerEntry;
typedef struct IMEISV_s{
  bstring   identity;
}IMEISV_t;

typedef struct{
  string mcc;
  string mnc;
}nas_plmn_t;

typedef struct{
  uint8_t type;
  std::vector<nas_plmn_t> plmn_list;
  std::vector<uint32_t> tac_list;
}p_tai_t;





}








#endif
