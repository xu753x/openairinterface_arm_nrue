#include "PlmnId.hpp"
#include "String2Value.hpp"

extern "C"{
  #include <math.h>
}
#include <iostream>
using namespace std;

namespace ngap{

	PlmnId::PlmnId(){}
	PlmnId::~PlmnId(){}

	void PlmnId::setMccMnc(const std::string mcc, const std::string mnc)
	{
		int mccValue = fromString<int>(mcc);
		int mncValue = fromString<int>(mnc);
		
		mcc_digit1 = mccValue/100;
		mcc_digit2 = (uint8_t)floor((double)(mccValue%100)/10);
		mcc_digit3 = mccValue%10;
		
		mnc_digit1 = (uint8_t)floor(((double)mncValue/100));
		if(!mnc_digit1) mnc_digit1 = 0xf;
		mnc_digit2 = (uint8_t)floor((double)(mncValue%100)/10);
		mnc_digit3 = mncValue%10;
	}
	void PlmnId::getMcc(std::string &mcc)
	{
		int m_mcc = mcc_digit1*100+mcc_digit2*10+mcc_digit3;
		mcc = to_string(m_mcc);
	}
	void PlmnId::getMnc(std::string &mnc)
	{
		int m_mnc = 0;
		if(mnc_digit1 == 0xf)
			m_mnc = mnc_digit2*10+mnc_digit3;
		else
			m_mnc = mnc_digit1*100+mnc_digit2*10+mnc_digit3;
		mnc = to_string(m_mnc);
	}

	bool PlmnId::encode2octetstring(Ngap_PLMNIdentity_t &plmn)
	{
		plmn.size = 3;//OCTET_STRING(SIZE(3))  9.3.3.5, 3gpp ts 38.413 V15.4.0
		uint8_t *buffer = (uint8_t*)calloc(1, 3*sizeof(uint8_t));
		if(!buffer) return false;
		
		buffer[0] = (mcc_digit1 & 0x0f) | ((mcc_digit2 & 0x0f)<<4);
		buffer[1] = (mcc_digit3 & 0x0f) | ((mnc_digit1 & 0x0f)<<4);
		buffer[2] = (mnc_digit2 & 0x0f) | ((mnc_digit3 & 0x0f)<<4);
		plmn.buf = buffer;

		return true;
	}

	bool PlmnId::decodefromoctetstring(Ngap_PLMNIdentity_t &plmn)
	{
		if(!plmn.buf) return false;
		mcc_digit1 = plmn.buf[0]&0x0f;
		mcc_digit2 = plmn.buf[0]>>4;
		mcc_digit3 = plmn.buf[1]&0x0f;
		mnc_digit1 = plmn.buf[1]>>4;
		mnc_digit2 = plmn.buf[2]&0x0f;
		mnc_digit3 = plmn.buf[2]>>4;
		
		return true;
	}

}
