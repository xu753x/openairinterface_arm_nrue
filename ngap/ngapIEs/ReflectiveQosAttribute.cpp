#include "ReflectiveQosAttribute.hpp"

#include <iostream>
using namespace std;

namespace ngap{

	ReflectiveQosAttribute::ReflectiveQosAttribute(){}  

	ReflectiveQosAttribute::~ReflectiveQosAttribute(){} 

	void ReflectiveQosAttribute::setReflectiveQosAttribute(e_Ngap_ReflectiveQosAttribute value)
	{
		attribute = value;
	}
	bool ReflectiveQosAttribute::getReflectiveQosAttribute(e_Ngap_ReflectiveQosAttribute &value)
	{
		if(attribute==0)
			value = (e_Ngap_ReflectiveQosAttribute)attribute;
		else
			return false;
		
		return true;
	}
	
	bool ReflectiveQosAttribute::encode2ReflectiveQosAttribute(Ngap_ReflectiveQosAttribute_t *reflectiveQosAttribute)
	{
		*reflectiveQosAttribute = attribute;
	
		return true;
	}
	bool ReflectiveQosAttribute::decodefromReflectiveQosAttribute(Ngap_ReflectiveQosAttribute_t *reflectiveQosAttribute)
	{
		attribute = *reflectiveQosAttribute;
		
		return true;
	}
}
