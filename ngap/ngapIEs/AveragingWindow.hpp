#ifndef _AVERAGINGWINDOW_H_
#define _AVERAGINGWINDOW_H_

extern "C"{
  #include "Ngap_AveragingWindow.h"
}

namespace ngap{

	class AveragingWindow{
	public:
		AveragingWindow();
		virtual ~AveragingWindow();
		
		void setAveragingWindow(long value);
		bool getAveragingWindow(long &value);
		
		bool encode2AveragingWindow(Ngap_AveragingWindow_t *);
		bool decodefromAveragingWindow(Ngap_AveragingWindow_t *);
	private:
		long averagingwindow;
	};

}

#endif


