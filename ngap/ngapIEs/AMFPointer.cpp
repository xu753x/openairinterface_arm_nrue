#include "AMFPointer.hpp"
#include "String2Value.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	AMFPointer::AMFPointer(){}  

	AMFPointer::~AMFPointer(){} 

	void AMFPointer::setAMFPointer(const std::string charPointer)
	{
		pointer = fromString<int>(charPointer);
	}
	void AMFPointer::getAMFPointer(std::string &charPointer)
	{
		 charPointer = to_string(pointer);
	}
	
	bool AMFPointer::encode2bitstring(Ngap_AMFPointer_t &amfpointer)
	{
		amfpointer.size = 1;
		uint8_t *buffer = (uint8_t *)calloc(1,sizeof(uint8_t));
		if(!buffer) return false;
		*buffer =((pointer & 0x3f)<<2);
		amfpointer.buf = buffer;
		amfpointer.bits_unused = 2;

		return true;
	}
	bool AMFPointer::decodefrombitstring(Ngap_AMFPointer_t &amfpointer)
	{
		if(!amfpointer.buf) return false;
		pointer = (amfpointer.buf[0] & 0xfc)>>2;//1111 1100

		return true;
	}
}

