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
cl_platform_id CreatePlatform();     // platform�� ����
void DisplayPlatformInfo(cl_platform_id id, cl_platform_info name, std::string str);    // platform�� ���� ���� ���

cl_device_id CreateDevice(cl_platform_id platform);     // device�� ����
void DisplayDeviceInfo(cl_device_id id);      // device�� ���� ���� ���

cl_context CreateContext(cl_platform_id platform, cl_device_id device);     // context�� ����

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device);      // command queue�� ����

cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);
cl_program CreateProgramFromBinary(cl_context context, cl_device_id device, const char* fileName);
bool SaveProgramBinary(cl_program program, cl_device_id device, const char* fileName);
