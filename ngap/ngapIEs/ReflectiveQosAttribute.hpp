#ifndef _REFLECTIVEQOSATTRIBUTE_H_
#define _REFLECTIVEQOSATTRIBUTE_H_

extern "C"{
  #include "Ngap_ReflectiveQosAttribute.h"
}

namespace ngap{

	class ReflectiveQosAttribute{
	public:
		ReflectiveQosAttribute();
		virtual ~ReflectiveQosAttribute();
		
		void setReflectiveQosAttribute(e_Ngap_ReflectiveQosAttribute value);
		bool getReflectiveQosAttribute(e_Ngap_ReflectiveQosAttribute &value);
		
		bool encode2ReflectiveQosAttribute(Ngap_ReflectiveQosAttribute_t *);
		bool decodefromReflectiveQosAttribute(Ngap_ReflectiveQosAttribute_t *);
	private:
		long attribute;
	};

}

#endif


