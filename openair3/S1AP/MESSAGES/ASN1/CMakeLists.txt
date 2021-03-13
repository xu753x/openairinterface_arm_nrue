file(GLOB lte_s1ap_source R15/s1ap-15.6.0.gen/*.c)
add_library(ASN1_S1AP_LIB ${lte_s1ap_source})
target_include_directories(ASN1_S1AP_LIB PUBLIC "R15/s1ap-15.6.0.gen/")
target_compile_options(ASN1_S1AP_LIB PRIVATE -DASN_DISABLE_OER_SUPPORT -w)
