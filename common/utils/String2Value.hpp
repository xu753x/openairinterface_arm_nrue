#include <iostream>
#include <sstream>
#include <string>
using namespace std;

template<class T>
inline T fromString(const string &str){
    istringstream is(str);              
    T v;                                     
    is>>v;                          
    return v;                
}
