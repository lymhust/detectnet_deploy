//============================================================================
// Name        : c_call_torch.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

extern "C"{
#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"
#include "luajit.h"
#include "luaT.h"
#include "TH/TH.h"
};

using namespace std;
using namespace cv;

/*
#include "luajit.h"
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
 */

void init_torch7(lua_State *L)
{
	// Load network
	if (luaL_loadfile(L, "./src/luafile/init_network.lua") || lua_pcall(L, 0, 0, 0))
	{
		printf("lua init error: %s \n", lua_tostring(L, -1));
	}

	// Load classify
	if (luaL_loadfile(L, "./src/luafile/classify.lua") || lua_pcall(L, 0, 0, 0))
	{
		printf("lua classify file load error: %s \n", lua_tostring(L, -1));
	}

	//luaT_newmetatable(L, "torch.FloatTensor", NULL, NULL, NULL, NULL);
	//luaT_newmetatable(L, "torch.IntTensor", NULL, NULL, NULL, NULL);

}

THFloatTensor* run_classify(lua_State *L, THFloatTensor *imgTensor)
{
	// Load the lua function
	lua_getglobal(L, "classify");
	if(!lua_isfunction(L,-1))
	{
		cout << "classify is not a function" << endl;
		lua_pop(L,1);
	}

	// This pushes data to the stack to be used as a parameter to the function call
	luaT_pushudata(L, (void *)imgTensor, "torch.FloatTensor");

	cout << "Number of stack before: " << lua_gettop(L) << endl;

	// Call the lua function
	if (lua_pcall(L, 1, 0, 0) != 0)
	{
		printf("lua error running function 'classify': %s \n", lua_tostring(L, -1));
	}
	
	// Get results returned from the lua function
	lua_getglobal(L, "bbox_nms");
	cout << "Number of stack after: " << lua_gettop(L) << endl;
	THFloatTensor *boxTensor = (THFloatTensor*)luaT_toudata(L, -1, "torch.FloatTensor");

	lua_settop(L, 0);
	return boxTensor;
}

int main()
{
	/*-------------------------------------------------------------*/
	// C call Torch7 init
	lua_State *L = luaL_newstate();
	luaL_openlibs(L);
	init_torch7(L);
	/*-------------------------------------------------------------*/
	
    namedWindow("Frame", CV_WINDOW_AUTOSIZE);
	for(int time=0; time<100; time++)
	{
		cout << time << endl;
		// system("pwd");

		clock_t start, end;
		
		Mat img = imread("./src/luafile/peds-007.png");
		
		if(img.empty())
		{
			cout<<"opencv error img is nill";
			return -1;
		}
		
		// Image resize and prepare
		start = clock();
		resize(img, img, Size(1024, 512), CV_INTER_LINEAR);
		img.convertTo(img, CV_32FC3);
		float *ptrimg = (float*)img.data; // image pointer
		
		/*-------------------------------------------------------------*/
		// C call Torch7
		// Convert the c array to Torch7 specific structure representing a tensor
		THFloatStorage *storage =  THFloatStorage_newWithData(ptrimg, img.rows * img.cols * img.channels());
		THFloatTensor *imgTensor = THFloatTensor_newWithStorage3d(storage, 0, img.rows, img.cols*img.channels(),       //long size0_, long stride0_,
				img.cols, img.channels(),
				img.channels(), 1);
		
		THFloatTensor *boxTensor = run_classify(L, imgTensor);	
		end = clock();
		cout << "Run time: " << (double)(end-start)/CLOCKS_PER_SEC*1000 << "ms" << endl;
		/*-------------------------------------------------------------*/
		
		// Get results
		img.convertTo(img, CV_8UC3);

		for (int i = 0; i < THFloatTensor_size(boxTensor, 0); i++) {
			float left = THFloatTensor_get2d(boxTensor, i, 0);
			float top = THFloatTensor_get2d(boxTensor, i, 1);
			float right = THFloatTensor_get2d(boxTensor, i, 2);
			float bottom = THFloatTensor_get2d(boxTensor, i, 3);
			float score = THFloatTensor_get2d(boxTensor, i, 4);
			char str_c[4];
			sprintf(str_c, "%.2f", score);
			string str(str_c);
			rectangle(img, Point(left, top), Point(right, bottom), Scalar(0,255,0), 2);
			putText(img, str, Point(left+2, top+15), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0,0,255), 2);
		}

		imshow("Frame", img);
		waitKey(1);
	}

	lua_close(L);
	return 0;
}


