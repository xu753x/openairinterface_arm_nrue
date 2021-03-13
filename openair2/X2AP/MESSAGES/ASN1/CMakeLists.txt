file(GLOB lte_x2ap_source R15/x2ap-15.6.0.gen/*.c)
add_library(ASN1_X2AP_LIB ${lte_x2ap_source})
target_include_directories(ASN1_X2AP_LIB PUBLIC "R15/x2ap-15.6.0.gen/")
target_compile_options(ASN1_X2AP_LIB PRIVATE -DASN_DISABLE_OER_SUPPORT -w)
