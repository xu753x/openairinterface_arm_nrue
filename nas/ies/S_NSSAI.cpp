#include "S_NSSAI.hpp"
#include "logger.hpp"
using namespace nas;


S_NSSAI::S_NSSAI(uint8_t iei) {
        _iei = iei;
}
S_NSSAI::S_NSSAI(const uint8_t iei, SNSSAI_s snssai) {
        _iei = iei;
		length = 3;
        SNSSAI = snssai;
		if (SNSSAI.sd != -1)
		{
			length += 3;
		}
		if (SNSSAI.mHplmnSst != -1)
		{
			length += 1;
		}
		if (SNSSAI.mHplmnSd != -1)
		{
			length += 3;
		};
}
S_NSSAI::S_NSSAI() {}
S_NSSAI::~S_NSSAI() {}

void S_NSSAI::setS_NSSAI(SNSSAI_s snssai) {
	SNSSAI = snssai;
}
void S_NSSAI::getValue(SNSSAI_s &snssai) {
	snssai=SNSSAI;
}

int S_NSSAI::encode2buffer(uint8_t *buf, int len) {
        Logger::nas_mm().debug("encoding S_NSSAI iei(0x%x)", _iei);
        if (len < length) {
                Logger::nas_mm().error("len is less than %d", length);
                return 0;
        }
        int encoded_size = 0;
        if (_iei) {
                *(buf + encoded_size) = _iei; encoded_size++;
                *(buf + encoded_size) = length - 2; encoded_size++;
                *(buf + encoded_size) = SNSSAI.sst;    encoded_size++;
				if (SNSSAI.sd != -1)
				{
					*(buf + encoded_size) = (SNSSAI.sd & 0x00ff0000) >> 16;    encoded_size++;
					*(buf + encoded_size) = (SNSSAI.sd & 0x0000ff00) >> 8;    encoded_size++;
					*(buf + encoded_size) = SNSSAI.sd & 0x000000ff;    encoded_size++;
				}
				if (SNSSAI.mHplmnSst != -1)
				{
					*(buf + encoded_size) = SNSSAI.mHplmnSst;    encoded_size++;
				}
				if (SNSSAI.mHplmnSd != -1)
				{
					*(buf + encoded_size) = (SNSSAI.mHplmnSd & 0x00ff0000) >> 16;    encoded_size++;
					*(buf + encoded_size) = (SNSSAI.mHplmnSd & 0x0000ff00) >> 8;    encoded_size++;
					*(buf + encoded_size) = SNSSAI.mHplmnSd & 0x000000ff;    encoded_size++;
				}
        }
        else {
           //     *(buf + encoded_size) = length - 1; encoded_size++;
             //   *(buf + encoded_size) = _value; encoded_size++; encoded_size++;
 }
        Logger::nas_mm().debug("encoded S_NSSAI len(%d)", encoded_size);
        return encoded_size;
}

int S_NSSAI::decodefrombuffer(uint8_t *buf, int len, bool is_option) {
        Logger::nas_mm().debug("decoding S_NSSAI iei(0x%x)", *buf);
        int decoded_size = 0;
		SNSSAI_s a={0,0,0,0};
        if (is_option) {
                decoded_size++;
}
        length = *(buf + decoded_size); decoded_size++;
		switch (*(buf + decoded_size-1))
		{
		case 1: {
			a.sst = *(buf + decoded_size); decoded_size++;//无 sd
			a.sd = -1;
			a.mHplmnSst = -1;
			a.mHplmnSd = -1;
		}break;
		case 4: {
			a.sst = *(buf + decoded_size); decoded_size++; 
			a.sd |= *(buf + decoded_size); decoded_size++; //有 sd
			a.sd << 8;
			a.sd |= *(buf + decoded_size); decoded_size++; //有 sd
			a.sd << 8;
			a.sd |= *(buf + decoded_size); decoded_size++; //有 sd
			a.mHplmnSst = -1;
			a.mHplmnSd = -1;
		}break;
		case 5: {
			a.sst = *(buf + decoded_size); decoded_size++; 
			a.sd |= *(buf + decoded_size); decoded_size++; //有 sd
			a.sd << 8;
			a.sd |= *(buf + decoded_size); decoded_size++; //有 sd
			a.sd << 8;
			a.sd |= *(buf + decoded_size); decoded_size++; //有 sd
			a.mHplmnSst = *(buf + decoded_size); decoded_size++; 
			a.mHplmnSd = -1;
		}break;
		case 8: {
			a.sst = *(buf + decoded_size); decoded_size++; 
			a.sd |= *(buf + decoded_size); decoded_size++; //有 sd
			a.sd << 8;
			a.sd |= *(buf + decoded_size); decoded_size++; //有 sd
			a.sd << 8;
			a.sd |= *(buf + decoded_size); decoded_size++; //有 sd
			a.mHplmnSst = *(buf + decoded_size); decoded_size++; 
			a.mHplmnSd |= *(buf + decoded_size); decoded_size++; //有 sd
			a.mHplmnSd << 16;
			a.mHplmnSd |= *(buf + decoded_size); decoded_size++; //有 sd
			a.mHplmnSd << 8;
			a.mHplmnSd |= *(buf + decoded_size); decoded_size++; //有 sd
		}break;
		}
		SNSSAI=a;
        Logger::nas_mm().debug("decoded S_NSSAI SST(0x%x) SD(0x%x) hplmnSST(0x%x) hplmnSD(0x%x)",a.sst, a.sd, a.mHplmnSst, a.mHplmnSd);
        Logger::nas_mm().debug("decoded S_NSSAI len(%d)", decoded_size);
        return decoded_size;
}



