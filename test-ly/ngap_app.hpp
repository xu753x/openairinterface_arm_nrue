#ifndef _NGAP_APPLICATION_H_
#define _NGAP_APPLICATION_H_

#include "sctp_server.hpp"
//#include "gNB_context.hpp"
#include <map>
#include <set>
#include <shared_mutex>
#include <string>
#include <thread>


using namespace sctp;

namespace ngap{

static const char * const ng_gnb_state_str [] = {"NGAP_INIT", "NGAP_RESETTING", "NGAP_READY", "NGAP_SHUTDOWN"};

class ngap_app : public sctp_application{
public:
  ngap_app(const string &address, const uint16_t port_num);
  ~ngap_app();
  uint32_t getPpid();
  void handle_receive(bstring payload, sctp_assoc_id_t assoc_id, sctp_stream_id_t stream, sctp_stream_id_t instreams, sctp_stream_id_t outstreams);
protected:
  sctp_server   sctp_s_38412;  
  uint32_t      ppid_;
 // std::map<sctp_assoc_id_t, std::shared_ptr<gnb_context>> assoc2gnbContext;
 // mutable std::shared_mutex m_assoc2gnbContext;

public:
 // void handle_sctp_new_association(sctp_assoc_id_t assoc_id, sctp_stream_id_t instreams, sctp_stream_id_t outstreams);
public:
 // bool is_assoc_id_2_gnb_context(const sctp_assoc_id_t & assoc_id) const;  
 // void set_assoc_id_2_gnb_context(const sctp_assoc_id_t& assoc_id, std::shared_ptr<gnb_context> gc);
 // std::shared_ptr<gnb_context>  assoc_id_2_gnb_context(const sctp_assoc_id_t & assoc_id) const;
};


}















#endif
