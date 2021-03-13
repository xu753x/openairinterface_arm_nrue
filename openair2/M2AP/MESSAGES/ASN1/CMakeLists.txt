file(GLOB lte_m2ap_source R14/m2ap-14.0.0.gen/*.c)
add_library(ASN1_M2AP_LIB ${lte_m2ap_source})
target_include_directories(ASN1_M2AP_LIB PUBLIC "R14/m2ap-14.0.0.gen/")
target_compile_options(ASN1_M2AP_LIB PRIVATE -DASN_DISABLE_OER_SUPPORT -w)
