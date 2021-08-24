#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>
#include <string>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

// constant
const int ARRAY_SIZE = 1024;

// functions
cl_platform_id CreatePlatform();     // platform을 생성
void DisplayPlatformInfo(cl_platform_id id, cl_platform_info name, std::string str);    // platform에 대한 정보 출력

cl_device_id CreateDevice(cl_platform_id platform);     // device를 생성
void DisplayDeviceInfo(cl_device_id id);      // device에 대한 정보 출력

cl_context CreateContext(cl_platform_id platform, cl_device_id device);     // context를 생성

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device);      // command queue를 생성

cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);
cl_program CreateProgramFromBinary(cl_context context, cl_device_id device, const char* fileName);
bool SaveProgramBinary(cl_program program, cl_device_id device, const char* fileName);
