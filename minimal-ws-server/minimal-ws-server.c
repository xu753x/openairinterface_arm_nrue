/*
 * lws-minimal-ws-server
 *
 * Written in 2010-2019 by Andy Green <andy@warmcat.com>
 *
 * This file is made available under the Creative Commons CC0 1.0
 * Universal Public Domain Dedication.
 *
 * This demonstrates the most minimal http server you can make with lws,
 * with an added websocket chat server.
 *
 * To keep it simple, it serves stuff in the subdirectory "./mount-origin" of
 * the directory it was started in.
 * You can change that by changing mount.origin.
 */

#include <libwebsockets.h>
#include <string.h>
#include <signal.h>
#include <time.h>
#include <signal.h>

#define HTTP_BUF 1024
volatile int  g_force_exit  = 0;

#ifdef WS_SERVER_ON
#include "openair2/LAYER2/NR_MAC_gNB/map.h"
#else
#include "../openair2/LAYER2/NR_MAC_gNB/map.h"
#endif

#define LWS_PLUGIN_STATIC
#include "protocol_lws_minimal.c"

int vrb_map_new[3][20][106];
int count;

int ue_speed_up[3];
int ue_speed_down[3];
int ue_state[3];
static struct lws_protocols protocols[] = {
	{ "http", lws_callback_http_dummy, 0, 0, 0, NULL, 0},
	LWS_PLUGIN_PROTOCOL_MINIMAL,
	LWS_PROTOCOL_LIST_TERM
};

static const lws_retry_bo_t retry = {
	.secs_since_valid_ping = 3,
	.secs_since_valid_hangup = 10,
};

static int interrupted;

static const struct lws_http_mount mount = {
	/* .mount_next */		NULL,		/* linked-list "next" */
	/* .mountpoint */		"/",		/* mountpoint URL */
	/* .origin */			"/home/witcomm/xw/code/ran2/minimal-ws-server/mount-origin",  /* serve from dir */
	/* .def */			"index.html",	/* default filename */
	/* .protocol */			NULL,
	/* .cgienv */			NULL,
	/* .extra_mimetypes */		NULL,
	/* .interpret */		NULL,
	/* .cgi_timeout */		0,
	/* .cache_max_age */		0,
	/* .auth_mask */		0,
	/* .cache_reusable */		0,
	/* .cache_revalidate */		0,
	/* .cache_intermediaries */	0,
	/* .origin_protocol */		LWSMPRO_FILE,	/* files in a dir */
	/* .mountpoint_len */		1,		/* char count */
	/* .basic_auth_login_file */	NULL,
};

void sigint_handler(int sig)
{
	interrupted = 1;
}

void sig_alarm_handler(int sig_num)
{
//    printf("%s, signal number:%d\n", __FUNCTION__, sig_num);
    if(sig_num = SIGALRM)
    {
        int remaing = alarm(1);

		for(int i=3;i<nf_status_arr_len;i++)
			nf_status_arr[i] = rand()%10;
#if 1
		int index=0;
		index=count-1;
		if(index==-1)
		  index=2;
		int j=0;
		int k=0;
		int tmp_data = 0;
		for(int i=3;i<nf_status_arr_len;i++)
		{
		   tmp_data = 0;
		   if(k==21)
		   {
			   tmp_data = vrb_map_new[index][j][105];
		   }
		   else
		   {
			   for(int tmp=0;tmp<5;tmp++)
			   {
				  if(tmp_data<vrb_map_new[index][j][k*5+tmp])
				  {
					tmp_data = vrb_map_new[index][j][k*5+tmp];
			      }
			   }
		   }
		   nf_status_arr[i]=tmp_data;
		   k=k+1;
           if(k==22)
		    {
			   k=0;
			   j=j+1;
			}
			
		}
		//memcpy(&nf_status_arr[2],&vrb_map_new[index],nf_status_arr_len);
		//memset(&vrb_map_new[index],0,nf_status_arr_len);
#endif 		
       update_client(nf_status_arr_len,nf_status_arr);
       
	//    int speed[8];
	//    speed[0] = 3;
    //    speed[1] = 6;
	//    for(int i = 1;i<4;i++){
	//       speed[i*2] = ue_speed_up[i-1];
	// 	  speed[i*2+1] = ue_speed_down[i-1];
	//    }
    //    update_client(8,speed);

	//    int ue_online[8];
	//    ue_online[0] = 4;
	//    ue_online[1] = 6;
	//    update_client(8,ue_online);
    }
}

#ifdef WS_SERVER_ON
int ws_server(int argc, const char **argv);

int ws_thread(int argc, const char **argv)
{
  pthread_t tids;
  int ret = pthread_create(&tids, NULL, ws_server, NULL);
  printf("pthread_create error: error_code= %d \n", ret);
}

int ws_server(int argc, const char **argv)
#else
int main(int argc, const char **argv)
#endif
{
	slices[0].ueid[0] = -1;
	slices[0].ueid[1] = -1;
	slices[0].ueid[2] = -1;
	slices[0].ueid[3] = -1;
	slices[0].ueid[4] = -1;
	slices[0].rbstartlocation = 25;
	slices[0].rboverlocation = 44;
    slices[0].slice_id = 1;
	for(int i = 0;i<16;i++){
       slices[0].slice_name[i] = i;
	}

	slices[1].ueid[0] = -1;
	slices[1].ueid[1] = -1;
	slices[1].ueid[2] = -1;
	slices[1].ueid[3] = -1;
	slices[1].ueid[4] = -1;
	slices[1].rbstartlocation = 45;
	slices[1].rboverlocation = 64;
    slices[1].slice_id = 2;
 	for(int i = 0;i<16;i++){
       slices[1].slice_name[i] = i;
	}

	slices[2].ueid[0] = -1;
	slices[2].ueid[1] = -1;
	slices[2].ueid[2] = -1;
	slices[2].ueid[3] = -1;
	slices[2].ueid[4] = -1;
	slices[2].rbstartlocation = 65;
	slices[2].rboverlocation = 84;
    slices[2].slice_id = 3;
	for(int i = 0;i<16;i++){
       slices[2].slice_name[i] = i;
	}
    slices[2].slice_online = 1;

	struct lws_context_creation_info info;
	struct lws_context *context;
	const char *p;
	int n = 0, logs = LLL_USER | LLL_ERR | LLL_WARN | LLL_NOTICE
			/* for LLL_ verbosity above NOTICE to be built into lws,
			 * lws must have been configured and built with
			 * -DCMAKE_BUILD_TYPE=DEBUG instead of =RELEASE */
			/* | LLL_INFO */ /* | LLL_PARSER */ /* | LLL_HEADER */
			/* | LLL_EXT */ /* | LLL_CLIENT */ /* | LLL_LATENCY */
			/* | LLL_DEBUG */;

//	signal(SIGINT, sigint_handler);
	signal(SIGALRM, sig_alarm_handler);
    alarm(1);

	nf_status_arr_len= RB_SIZE_deal*SLOT_IN_FRAME+3;
	nf_status_arr[0] = 2;
	nf_status_arr[1] = RB_SIZE_deal; //rb size
	nf_status_arr[2] = SLOT_IN_FRAME; //time 

	if ((p = lws_cmdline_option(argc, argv, "-d")))
		logs = atoi(p);
//    logs = 255;
	lws_set_log_level(logs, NULL);
	lwsl_user("LWS minimal ws server | visit http://localhost:7681 (-s = use TLS / https)\n");

	memset(&info, 0, sizeof info); /* otherwise uninitialized garbage */
	info.port = 7681;
	info.mounts = &mount;
	info.protocols = protocols;
	info.vhost_name = "localhost";
	info.options =
		LWS_SERVER_OPTION_HTTP_HEADERS_SECURITY_BEST_PRACTICES_ENFORCE;

#if defined(LWS_WITH_TLS)
	if (lws_cmdline_option(argc, argv, "-s")) {
		lwsl_user("Server using TLS\n");
		info.options |= LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT;
		info.ssl_cert_filepath = "localhost-100y.cert";
		info.ssl_private_key_filepath = "localhost-100y.key";
	}
#endif

	if (lws_cmdline_option(argc, argv, "-h"))
		info.options |= LWS_SERVER_OPTION_VHOST_UPG_STRICT_HOST_CHECK;

	if (lws_cmdline_option(argc, argv, "-v"))
		info.retry_and_idle_policy = &retry;

	context = lws_create_context(&info);
	if (!context) {
		lwsl_err("lws init failed\n");
		return 1;
	}

	while (n >= 0 && !interrupted)
		n = lws_service(context, 0);

	lws_context_destroy(context);

	return 0;
}



