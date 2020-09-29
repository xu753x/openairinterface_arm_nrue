#ifndef __UES_Usage_Setting_H_
#define __UES_Usage_Setting_H_

#include <stdint.h>

namespace nas {

	class UES_Usage_Setting {
	public:
		UES_Usage_Setting();
		UES_Usage_Setting(uint8_t iei);
		UES_Usage_Setting(const uint8_t iei, bool ues_usage_setting);
		~UES_Usage_Setting();
		int encode2buffer(uint8_t *buf, int len);
		int decodefrombuffer(uint8_t *buf, int len, bool is_option);
		void setValue(bool value);
		bool getValue();
	private:
		uint8_t _iei;
		bool _ues_usage_setting;
	};
}














#endif

