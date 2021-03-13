file(GLOB lte_f1ap_source R15.2.1/f1ap-15.2.1.gen/*.c)
add_library(ASN1_F1AP_LIB ${lte_f1ap_source})
target_include_directories(ASN1_F1AP_LIB PUBLIC "R15.2.1/f1ap-15.2.1.gen/")
target_compile_options(ASN1_F1AP_LIB PRIVATE -DASN_DISABLE_OER_SUPPORT -w)
