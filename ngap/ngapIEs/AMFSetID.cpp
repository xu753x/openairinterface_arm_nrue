#include "AMFSetID.hpp"
#include "String2Value.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	AMFSetID::AMFSetID(){}  

	AMFSetID::~AMFSetID(){} 

	void AMFSetID::setAMFSetID(const std::string charid)
	{
    	setid = fromString<uint16_t>(charid);
	}
	void AMFSetID::getAMFSetID(std::string &charid)
	{
		charid = to_string(setid);
	}
	
	bool AMFSetID::encode2bitstring(Ngap_AMFSetID_t &amfsetid)
	{
		amfsetid.size = 2;
		uint8_t *buffer = (uint8_t *)calloc(1,sizeof(uint16_t));
		if(!buffer) return false;
		//*(uint16_t *)buffer = setid & 0x3ff; 
                buffer[0] = ((setid & 0x03fc)>>2);
                buffer[1] = ((setid & 0x0003)<<6);
		amfsetid.buf = buffer;
		amfsetid.bits_unused = 6;
		
		return true;
	}
	bool AMFSetID::decodefrombitstring(Ngap_AMFSetID_t &amfsetid)
	{
		if(!amfsetid.buf) return false;
                printf("test for amfsetid\n");
                for(int i=0; i<amfsetid.size; i++){
                  printf("%x ", amfsetid.buf[i]);
                }
                printf("\n");
                uint16_t temp = 0;
                temp |= amfsetid.buf[0]<<8;
                temp |= amfsetid.buf[1];
		setid = (temp&0xffc0)>>6;//1111 1111 11 00 0000

		return true;
	}
}

