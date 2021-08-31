/*
 * ws protocol handler plugin for "lws-minimal"
 *
 * Written in 2010-2019 by Andy Green <andy@warmcat.com>
 *
 * This file is made available under the Creative Commons CC0 1.0
 * Universal Public Domain Dedication.
 *
 * This version holds a single message at a time, which may be lost if a new
 * message comes.  See the minimal-ws-server-ring sample for the same thing
 * but using an lws_ring ringbuffer to hold up to 8 messages at a time.
 */

#if !defined (LWS_PLUGIN_STATIC)
#define LWS_DLL
#define LWS_INTERNAL
#include <libwebsockets.h>
#endif

#include <string.h>

/* one of these created for each message */

struct msg {
	void *payload; /* is malloc'd */
	size_t len;
};

/* one of these is created for each client connecting to us */

struct per_session_data__minimal {
	struct per_session_data__minimal *pss_list;
	struct lws *wsi;
	int last; /* the last message number we sent */
};

/* one of these is created for each vhost our protocol is used with */

struct per_vhost_data__minimal {
	struct lws_context *context;
	struct lws_vhost *vhost;
	const struct lws_protocols *protocol;

	struct per_session_data__minimal *pss_list; /* linked-list of live pss*/

	struct msg amsg; /* the one pending message... */
	int current; /* the current message number we are caching */
};

/* destroys the message when everyone has had a copy of it */
int update_client(int len, const char* payload);

static void
__minimal_destroy_message(void *_msg)
{
	struct msg *msg = _msg;

	free(msg->payload);
	msg->payload = NULL;
	msg->len = 0;
}

static int first_conn = 0;
static int notifying = 0;
struct lws *wsi_client;


#define RB_SIZE 		20
#define MS_IN_FRAME 	10
#define SLOT_IN_MS 		2
#define SLOT_IN_FRAME 	(MS_IN_FRAME*SLOT_IN_MS)

char nf_status_arr[10000]={0};
int nf_status_arr_len=0;

int update_client(int len, const char* payload)
{	
    char sendArr[10000];
	if (notifying != 0)
		return 0;
    if (len > 10000)
		return 0;
    if (wsi_client == NULL)
		return 0;

    notifying = 1;
	lwsl_user("update_client  size %d. %p.\n",len,payload);

	memcpy(sendArr + LWS_PRE, payload, len);
	/* notice we allowed for LWS_PRE in the payload already */
	int m = lws_write(wsi_client, sendArr + LWS_PRE, len, LWS_WRITE_TEXT);
	if (m < len) {
		lwsl_err("ERROR %d writing to ws\n", m);
	}
	notifying = 0;
    return (m-len);
}

static int
callback_minimal(struct lws *wsi, enum lws_callback_reasons reason,
			void *user, void *in, size_t len)
{
	struct per_session_data__minimal *pss =
			(struct per_session_data__minimal *)user;
	struct per_vhost_data__minimal *vhd =
			(struct per_vhost_data__minimal *)
			lws_protocol_vh_priv_get(lws_get_vhost(wsi),
					lws_get_protocol(wsi));
	int m;

	switch (reason) {
	case LWS_CALLBACK_PROTOCOL_INIT:
		lwsl_user("LWS_CALLBACK_PROTOCOL_INIT \n");
		vhd = lws_protocol_vh_priv_zalloc(lws_get_vhost(wsi),
				lws_get_protocol(wsi),
				sizeof(struct per_vhost_data__minimal));
		vhd->context = lws_get_context(wsi);
		vhd->protocol = lws_get_protocol(wsi);
		vhd->vhost = lws_get_vhost(wsi);
		break;

	case LWS_CALLBACK_ESTABLISHED:
		lwsl_user("LWS_CALLBACK_ESTABLISHED \n");
		/* add ourselves to the list of live pss held in the vhd */
		lws_ll_fwd_insert(pss, pss_list, vhd->pss_list);
		pss->wsi = wsi;
		pss->last = vhd->current;
		wsi_client = wsi;
		first_conn = 1;
		break;

	case LWS_CALLBACK_CLOSED:
		lwsl_user("LWS_CALLBACK_CLOSED \n");
		/* remove our closing pss from the list of live pss */
		lws_ll_fwd_remove(struct per_session_data__minimal, pss_list,
				  pss, vhd->pss_list);
		wsi_client = NULL;
		break;

	case LWS_CALLBACK_SERVER_WRITEABLE:
		lwsl_user("LWS_CALLBACK_SERVER_WRITEABLE len %d %p \n",vhd->amsg.len,vhd->amsg.payload);
		if (!vhd->amsg.payload)
		{
			if (1 == first_conn)
			{
				update_client(nf_status_arr_len,nf_status_arr);
				first_conn = 0;
			}
			break;
		}

		// if (pss->last == vhd->current)
		// 	break;

		/* notice we allowed for LWS_PRE in the payload already */
		m = lws_write(wsi, ((unsigned char *)vhd->amsg.payload) +
			      LWS_PRE, vhd->amsg.len, LWS_WRITE_TEXT);
		if (m < (int)vhd->amsg.len) {
			lwsl_err("ERROR %d writing to ws\n", m);
			return -1;
		}

		pss->last = vhd->current;
		break;

	case LWS_CALLBACK_RECEIVE:
		lwsl_user("LWS_CALLBACK_RECEIVE len %d \n",vhd->amsg.len);
		if (vhd->amsg.payload)
			__minimal_destroy_message(&vhd->amsg);

		vhd->amsg.len = len;
		/* notice we over-allocate by LWS_PRE */
		vhd->amsg.payload = malloc(LWS_PRE + len);
		if (!vhd->amsg.payload) {
			lwsl_user("OOM: dropping\n");
			break;
		}

		memcpy((char *)vhd->amsg.payload + LWS_PRE, in, len);
		vhd->current++;

		update_client(vhd->amsg.len,in);
		// /*
		//  * let everybody know we want to write something on them
		//  * as soon as they are ready
		//  */
		// lws_start_foreach_llp(struct per_session_data__minimal **,
		// 		      ppss, vhd->pss_list) {
		// 	lws_callback_on_writable((*ppss)->wsi);
		// } lws_end_foreach_llp(ppss, pss_list);
		break;

	default:
		break;
	}

	return 0;
}

#define LWS_PLUGIN_PROTOCOL_MINIMAL \
	{ \
		"lws-minimal", \
		callback_minimal, \
		sizeof(struct per_session_data__minimal), \
		128, \
		0, NULL, 0 \
	}
