#include "sctp_server.hpp"
#include "NGSetupRequest.hpp"
extern "C"{
  #include "Ngap_NGAP-PDU.h"
}


using namespace sctp;
using namespace ngap;


typedef int (*ngap_message_decoded_callback)(
    const sctp_assoc_id_t             assoc_id,
    const sctp_stream_id_t            stream,
    struct Ngap_NGAP_PDU *message_p
);
int ngap_amf_handle_ng_setup_request(const sctp_assoc_id_t             assoc_id,  const sctp_stream_id_t            stream,  struct Ngap_NGAP_PDU *message_p){
  printf("sending itti ng setup request message to TASK_AMF_N2");
  return 0;
}
int ngap_amf_handle_ng_setup_response(const sctp_assoc_id_t             assoc_id,  const sctp_stream_id_t            stream,  struct Ngap_NGAP_PDU *message_p){
  printf("sending itti ng setup response message to TASK_AMF_N2");
  return 0;
}

int ngap_amf_handle_initial_ue_message(const sctp_assoc_id_t             assoc_id,  const sctp_stream_id_t            stream,  struct Ngap_NGAP_PDU *message_p){
  printf("sending itti initial ue message to TASK_AMF_N2");
  return 0;
}
int ngap_amf_handle_uplink_nas_transport(const sctp_assoc_id_t             assoc_id,  const sctp_stream_id_t            stream,  struct Ngap_NGAP_PDU *message_p){
	printf("sending itti up link nas transport message to TASK_AMF_N2");
	return 0;
}
int ngap_amf_handle_initial_context_setup_response(const sctp_assoc_id_t             assoc_id,  const sctp_stream_id_t            stream,  struct Ngap_NGAP_PDU *message_p){
  printf("sending itti initial context setup response to TASK_AMF_N11");
  return 0;
}
int ngap_amf_handle_initial_context_setup_failure(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU *message_p){
  printf("sending itti initial context setup failure to TASK_AMF_N2");
  return 0;
}
int ngap_amf_handle_ue_radio_cap_indication(const sctp_assoc_id_t             assoc_id,  const sctp_stream_id_t            stream,  struct Ngap_NGAP_PDU *message_p){
  printf("sending itti ue radio capability indication to TASK_AMF_N2");
  return 0;
}
int ngap_amf_handle_ue_context_release_request(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU *message_p){
  printf("sending itti ue context release request to TASK_AMF_N2");
  return 0;
}
int ngap_amf_handle_ue_context_release_complete(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU *message_p){
  printf("sending itti ue context release complete to TASK_AMF_N2");
  return 0;
}
int ngap_amf_handle_pdu_session_resource_setup_response(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU *message_p){
  printf("sending itti pdu_session_resource_setup_response to TASK_AMF_N11");
  return 0;
}
int ngap_amf_handle_pdu_session_resource_release_response(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU *message_p) {
	printf("sending itti pdu session resource release response to TASK_AMF_N2");
	return 0;
}

int ngap_amf_handle_error_indication(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU *message_p) {
	printf("sending itti ngap error indication to TASK_AMF_N2");
	return 0;
}
int ngap_amf_configuration_update(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU *message_p) {
	printf("sending itti amf configuration update to TASK_AMF_N2");
	return 0;
}
int amf_status_indication (const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU *message_p) {
        printf("sending itti amf status indication to TASK_AMF_N2");
        return 0;
}
int cell_traffic_trace (const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU *message_p) {
        printf("sending itti cell_traffic_trace to TASK_AMF_N2");
        return 0;
}
int deactivate_trace (const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU *message_p) {
        printf("sending itti deactivate trace to TASK_AMF_N2");
        return 0;
}
int downlink_nas_transport (const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU *message_p) {
        printf("sending itti downlink nas transport to TASK_AMF_N2");
        return 0;
}
int downlink_non_UEassociated_nrppa_transport(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti downlink non UEassociated nrppa transport to TASK_AMF_N2");
    return 0;
}
int downlink_ran_configuration_transfer(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti downlink ran configuration transfer to TASK_AMF_N2");
    return 0;
}
int downlink_ran_status_transfer(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti downlink ran status transfer to TASK_AMF_N2");
    return 0;
}
int downlink_ue_associated_nappa_transport(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti downlink ue associated nappa transport to TASK_AMF_N2");
    return 0;
}
int handover_cancel(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti handover cancel to TASK_AMF_N2");
    return 0;
}
int handover_notification(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti handover notification to TASK_AMF_N2");
    return 0;
}
int handover_preparation(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti handover preparation to TASK_AMF_N2");
    return 0;
}
int handover_resource_allocation(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti handover resource allocation to TASK_AMF_N2");
    return 0;
}
int location_reporting_control(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti location reporting control to TASK_AMF_N2");
    return 0;
}
int location_reporting_failure_indication(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti location reporting failure indication to TASK_AMF_N2");
    return 0;
}
int location_report(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti location report to TASK_AMF_N2");
    return 0;
}
int nas_non_delivery_indication(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti nas non delivery indication to TASK_AMF_N2");
    return 0;
}
int ng_reset(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti ng reset to TASK_AMF_N2");
    return 0;
}
int overload_start(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti overload start to TASK_AMF_N2");
    return 0;
}
int overload_stop(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti overload stop to TASK_AMF_N2");
    return 0;
}
int paging(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti paging to TASK_AMF_N2");
    return 0;
}
int ngap_amf_handle_path_switch_request(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti ngap amf handle path switch request to TASK_AMF_N2");
    return 0;
}
int pdu_session_resource_modify(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti pdu session resource modify to TASK_AMF_N2");
    return 0;
}
int pdu_session_resource_modify_indication(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti pdu session resource modify indication to TASK_AMF_N2");
    return 0;
}
int pdu_session_resource_notify(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti pdu session resource notify to TASK_AMF_N2");
    return 0;
}
int private_message(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti private message to TASK_AMF_N2");
    return 0;
}
int pws_cancel(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti pws cancel to TASK_AMF_N2");
    return 0;
}
int pws_failure_indication(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti pws failure indication to TASK_AMF_N2");
    return 0;
}
int pws_restart_indication(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti pws restart indication to TASK_AMF_N2");
    return 0;
}
int ran_configuration_update(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti ran configuration update to TASK_AMF_N2");
    return 0;
}
int reroute_nas_request(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti reroute nas request to TASK_AMF_N2");
    return 0;
}
int rrc_inactive_transition_report(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti rrc inactive transition report to TASK_AMF_N2");
    return 0;
}
int trace_failure_indication(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti trace failure indication to TASK_AMF_N2");
    return 0;
}
int trace_start(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti trace start to TASK_AMF_N2");
    return 0;
}
int ue_context_modification(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti ue context modification to TASK_AMF_N2");
    return 0;
}
int ue_radio_capability_check(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti ue radio capability check to TASK_AMF_N2");
    return 0;
}
int ue_tnla_binding_release(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti ue tnla binding release to TASK_AMF_N2");
    return 0;
}
int uplink_non_ue_associated_nrppa_transport(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti uplink non ue associated nrppa transport to TASK_AMF_N2");
    return 0;
}
int uplink_ran_configuration_transfer(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti uplink ran configuration transfer to TASK_AMF_N2");
    return 0;
}

int uplink_ran_status_transfer(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti uplink ran status transfer to TASK_AMF_N2");
    return 0;
}

int uplink_ue_associated_nrppa_transport(const sctp_assoc_id_t assoc_id, const sctp_stream_id_t stream, struct Ngap_NGAP_PDU* message_p) {
    printf("sending itti uplink ue associated nrppa transport to TASK_AMF_N2");
    return 0;
}




ngap_message_decoded_callback   ngap_messages_callback[][3] = {
    {ngap_amf_configuration_update,ngap_amf_configuration_update,ngap_amf_configuration_update}, /*0 AMFConfigurationUpdate*/
    {amf_status_indication,amf_status_indication,amf_status_indication}, /*1 AMFStatusIndication*/
    {cell_traffic_trace,cell_traffic_trace,cell_traffic_trace}, /*2 CellTrafficTrace*/
    {deactivate_trace,deactivate_trace,deactivate_trace}, /*3 DeactivateTrace*/
    {downlink_nas_transport,downlink_nas_transport,downlink_nas_transport}, /*4 DownlinkNASTransport*/
    {downlink_non_UEassociated_nrppa_transport,downlink_non_UEassociated_nrppa_transport,downlink_non_UEassociated_nrppa_transport}, /*5 DownlinkNonUEAssociatedNRPPaTransport*/
    {downlink_ran_configuration_transfer,downlink_ran_configuration_transfer,downlink_ran_configuration_transfer}, /*6 DownlinkRANConfigurationTransfer*/
    {downlink_ran_status_transfer,downlink_ran_status_transfer,downlink_ran_status_transfer}, /*7 DownlinkRANStatusTransfer*/
    {downlink_ue_associated_nappa_transport,downlink_ue_associated_nappa_transport,downlink_ue_associated_nappa_transport}, /*8 DownlinkUEAssociatedNRPPaTransport*/
    {ngap_amf_handle_error_indication,0,0},//9 {ngap_amf_handle_error_indication,0,0}, /*ErrorIndication*/
    {handover_cancel,handover_cancel,handover_cancel}, /*10 HandoverCancel*/
    {handover_notification,handover_notification,handover_notification}, /*11 HandoverNotification*/
    {handover_preparation,handover_preparation,handover_preparation}, /*12 HandoverPreparation*/
    {handover_resource_allocation,handover_resource_allocation,handover_resource_allocation}, /*13 HandoverResourceAllocation*/
    {0,ngap_amf_handle_initial_context_setup_response,ngap_amf_handle_initial_context_setup_failure}, /*InitialContextSetup*/
    {ngap_amf_handle_initial_ue_message,0,0},//15 {ngap_amf_handle_initial_ue_message,0,0}, /*InitialUEMessage*/
    {location_reporting_control,location_reporting_control,location_reporting_control}, /*16 LocationReportingControl*/
    {location_reporting_failure_indication,location_reporting_failure_indication,location_reporting_failure_indication}, /*17 LocationReportingFailureIndication*/
    {location_report,location_report,location_report}, /*18 LocationReport*/
    {nas_non_delivery_indication,nas_non_delivery_indication,nas_non_delivery_indication}, /*19 NASNonDeliveryIndication*/
    {ng_reset,ng_reset,ng_reset}, /*20 NGReset*/
    {ngap_amf_handle_ng_setup_request,ngap_amf_handle_ng_setup_response,ngap_amf_handle_ng_setup_request}, /*21 NGSetup*/
    {overload_start,overload_start,overload_start}, /*OverloadStart*/
    {overload_stop,overload_stop,overload_stop}, /*OverloadStop*/
    {paging,paging,paging}, /*Paging*/
    {ngap_amf_handle_path_switch_request,ngap_amf_handle_path_switch_request,ngap_amf_handle_path_switch_request},//{ngap_amf_handle_path_switch_request,0,0}, /*PathSwitchRequest*
    {pdu_session_resource_modify,pdu_session_resource_modify,pdu_session_resource_modify}, /*PDUSessionResourceModify*/
    {pdu_session_resource_modify_indication,pdu_session_resource_modify_indication,pdu_session_resource_modify_indication}, /*PDUSessionResourceModifyIndication*/
    {0,ngap_amf_handle_pdu_session_resource_release_response,0}, /*PDUSessionResourceRelease*/
    {0,ngap_amf_handle_pdu_session_resource_setup_response,0}, /*PDUSessionResourceSetup*/
    {pdu_session_resource_notify,pdu_session_resource_notify,pdu_session_resource_notify}, /*PDUSessionResourceNotify*/
    {private_message,private_message,private_message}, /*PrivateMessage*/
    {pws_cancel,pws_cancel,pws_cancel}, /*PWSCancel*/
    {pws_failure_indication,pws_failure_indication,pws_failure_indication}, /*PWSFailureIndication*/
    {pws_restart_indication,pws_restart_indication,pws_restart_indication}, /*PWSRestartIndication*/
    {ran_configuration_update,ran_configuration_update,ran_configuration_update}, /*RANConfigurationUpdate*/
    {reroute_nas_request,reroute_nas_request,reroute_nas_request}, /*RerouteNASRequest*/
    {rrc_inactive_transition_report,rrc_inactive_transition_report,rrc_inactive_transition_report}, /*RRCInactiveTransitionReport*/
    {trace_failure_indication,trace_failure_indication,trace_failure_indication}, /*TraceFailureIndication*/
    {trace_start,trace_start,trace_start}, /*TraceStart*/
    {ue_context_modification,ue_context_modification,ue_context_modification}, /*UEContextModification*/
    {0,ngap_amf_handle_ue_context_release_complete,0}, /*UEContextRelease*/
    {ngap_amf_handle_ue_context_release_request,0,0}, /*UEContextReleaseRequest*/
    {ue_radio_capability_check,ue_radio_capability_check,ue_radio_capability_check}, /*UERadioCapabilityCheck*/
    {ngap_amf_handle_ue_radio_cap_indication,0,0}, /*UERadioCapabilityInfoIndication*/
    {ue_tnla_binding_release,ue_tnla_binding_release,ue_tnla_binding_release}, /*UETNLABindingRelease*/
    {ngap_amf_handle_uplink_nas_transport,0,0},//{ngap_amf_handle_uplink_nas_transport,0,0}, /*UplinkNASTransport*/
    {uplink_non_ue_associated_nrppa_transport,uplink_non_ue_associated_nrppa_transport,uplink_non_ue_associated_nrppa_transport}, /*UplinkNonUEAssociatedNRPPaTransport*/
    {uplink_ran_configuration_transfer,uplink_ran_configuration_transfer,uplink_ran_configuration_transfer}, /*UplinkRANConfigurationTransfer*/
    {uplink_ran_status_transfer,uplink_ran_status_transfer,uplink_ran_status_transfer}, /*UplinkRANStatusTransfer*/
    {uplink_ue_associated_nrppa_transport,uplink_ue_associated_nrppa_transport,uplink_ue_associated_nrppa_transport}, /*UplinkUEAssociatedNRPPaTransport*/
    {0,0,0}, /*WriteReplaceWarning*/
    {0,0,0}, /*WriteReplaceWarning*/
    {0,0,0}, /*WriteReplaceWarning*/
    {0,0,0}, /*WriteReplaceWarning*/
    {0,0,0}, /*WriteReplaceWarning*/
    {0,0,0}, /*WriteReplaceWarning*/
    {0,0,0}, /*WriteReplaceWarning*/
    {0,0,0}, /*WriteReplaceWarning*/
    {0,0,0} /*WriteReplaceWarning*/
};
