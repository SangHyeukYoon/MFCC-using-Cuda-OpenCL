#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <thread>
#include <chrono>
#include <cmath>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "clUtils.h"
#include "Wav.h"

#define PROFILING

constexpr char FILE_PATH[] = "C:\\Users\\nyoon\\Music\\wav\\s1_anger_M_a1.wav";

constexpr float PRE_EMPHASIS = 0.97;
constexpr float FRAME_SIZE = 0.032;
constexpr float FRAME_STRIDE = 0.016;
constexpr int BIT_LENGTH = 9;
constexpr int NFFT = 1 << BIT_LENGTH;
constexpr int NFILTER = 40;
constexpr int NUM_CEPS = 12;
constexpr float CEP_LIFTER = 23.0;

template <typename T>
void PrintArray(int length, T arr)
{
    for (int i = 0; i < length; ++i)
    {
        std::cout << arr[i] << ", ";

        if ((i + 1) % 8 == 0)
        {
            std::cout << std::endl;
        }
    }

    std::cout << std::endl;
}

int main(int argc, const char* argv[])
{
    // constants
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context context = 0;
    cl_command_queue commandQueue;
    cl_program program = 0;
    cl_int errNum;

    const char* clPath = "mfcc.cl";

    //--------------------------------//
    //          Set up section        //
    //--------------------------------//

    // crate platform
    platform = CreatePlatform();
    if (platform == NULL)
    {
        std::cerr << "Failed to create OpenCL platform." << std::endl;

        return 1;
    }

    // create device
    device = CreateDevice(platform);
    if (device == NULL)
    {
        std::cerr << "Failed to create OpenCL device." << std::endl;

        return 1;
    }

    // create context
    context = CreateContext(platform, device);
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;

        return 1;
    }

    // create command queue
    commandQueue = CreateCommandQueue(context, device);
    if (commandQueue == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;

        return 1;
    }

    // create program
    program = CreateProgram(context, device, clPath);
    if (program == NULL)
    {
        std::cerr << "Failed to create OpenCL program." << std::endl;

        return 1;
    }

    //--------------------------------//
    //           Create datas         //
    //--------------------------------//

    Wav wav{ FILE_PATH };

    short* signal = wav.GetData();
    const int signalLen = wav.GetLen();
    const int sr = 16000;

    int frameLength = FRAME_SIZE * sr;
    int frameStep = FRAME_STRIDE * sr;
    int numFrames = ceil(signalLen / frameStep);

    //----------------------------------//
    //           Pre-Processing         //
    //----------------------------------//

    // create buffers
    cl_mem signal_mem, frames_mem;

    signal_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(short) * signalLen, nullptr, &errNum);
    frames_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * numFrames * frameLength, nullptr, &errNum);

    if (signal_mem == nullptr || frames_mem == nullptr)
    {
        std::cerr << "Failed to create signal_mem or frames_mem buffers." << std::endl;

        return 1;
    }

    // create kernel
    cl_kernel pre_ker = clCreateKernel(program, "PreProcessing", &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create PreProcessing kernel." << std::endl;

        return 1;
    }

    // set kernel argument
    cl_uint argCount = 0;
    errNum = clSetKernelArg(pre_ker, argCount++, sizeof(float), &PRE_EMPHASIS);
    errNum |= clSetKernelArg(pre_ker, argCount++, sizeof(int), &frameLength);
    errNum |= clSetKernelArg(pre_ker, argCount++, sizeof(int), &frameStep);
    errNum |= clSetKernelArg(pre_ker, argCount++, sizeof(int), &signalLen);
    errNum |= clSetKernelArg(pre_ker, argCount++, sizeof(cl_mem), &signal_mem);
    errNum |= clSetKernelArg(pre_ker, argCount++, sizeof(cl_mem), &frames_mem);
    errNum |= clSetKernelArg(pre_ker, argCount++, sizeof(short) * (frameLength + 2), nullptr);
    errNum |= clSetKernelArg(pre_ker, argCount++, sizeof(float) * frameLength, nullptr);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create PreProcessing kernel argument." << std::endl;

        return 1;
    }

    // execute kernel arguments
    cl_uint nd_pre = 2;
    size_t global_pre[] = { numFrames, frameLength };
    size_t local_pre[] = { 1, frameLength };

    //----------------------------------//
    //      FFT and Power Spectrum      //
    //----------------------------------//

    // create buffers
    cl_mem powFrames_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * numFrames * (NFFT / 2 + 1), nullptr, &errNum);

    if (powFrames_mem == nullptr)
    {
        std::cerr << "Failed to create powFrames_mem buffers." << std::endl;

        return 1;
    }

    // create kernel
    cl_kernel powFFT_ker = clCreateKernel(program, "PowerFFT", &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create PowerFFT kernel." << std::endl;

        return 1;
    }

    // set kernel argument
    argCount = 0;
    errNum = clSetKernelArg(powFFT_ker, argCount++, sizeof(int), &BIT_LENGTH);
    errNum |= clSetKernelArg(powFFT_ker, argCount++, sizeof(int), &frameLength);
    errNum |= clSetKernelArg(powFFT_ker, argCount++, sizeof(cl_mem), &frames_mem);
    errNum |= clSetKernelArg(powFFT_ker, argCount++, sizeof(cl_mem), &powFrames_mem);
    errNum |= clSetKernelArg(powFFT_ker, argCount++, sizeof(cl_float2) * NFFT, nullptr);
    errNum |= clSetKernelArg(powFFT_ker, argCount++, sizeof(cl_float2) * NFFT, nullptr);
    errNum |= clSetKernelArg(powFFT_ker, argCount++, sizeof(cl_float2) * (NFFT + 2), nullptr);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create PowerFFT kernel argument." << std::endl;

        return 1;
    }

    // execute kernel arguments
    cl_uint nd_fft = 1;
    size_t global_fft[] = { static_cast<size_t>(numFrames * NFFT / 2) };
    size_t local_fft[] = { static_cast<size_t>(NFFT / 2) };

    //----------------------------------//
    //           Filter Banks           //
    //----------------------------------//

    float lowFreqMel = 0;
    float highFreqMel = 2595 * log10(1 + (sr / 2) / 700);
    float melPoints[NFILTER + 2];

    float melStride = (highFreqMel - lowFreqMel) / (NFILTER + 1);
    for (int i = 0; i < NFILTER + 2; ++i)
    {
        melPoints[i] = lowFreqMel + melStride * i;
    }

    float melBins[NFILTER + 2];
    for (int i = 0; i < NFILTER + 2; ++i)
    {
        melBins[i] = floor((NFFT + 1) * (700 * (pow(10, (melPoints[i] / 2595)) - 1)) / sr);
    }

    const int fftResultLen = NFFT / 2 + 1;

    float* fbank = new float[NFILTER * fftResultLen];
    memset(fbank, 0.0, sizeof(float) * (NFILTER * fftResultLen));

    float f_m_minus, f_m, f_m_plus;
    for (int m = 1; m <= NFILTER; ++m)
    {
        f_m_minus = melBins[m - 1];
        f_m = melBins[m];
        f_m_plus = melBins[m + 1];

        for (int k = f_m_minus; k < f_m; ++k)
        {
            fbank[(m - 1) * fftResultLen + k] = (k - f_m_minus) / (f_m - f_m_minus);
        }

        for (int k = f_m; k < f_m_plus; ++k)
        {
            fbank[(m - 1) * fftResultLen + k] = (f_m_plus - k) / (f_m_plus - f_m);
        }
    }

    float* fbank_T = new float[fftResultLen * NFILTER];
    for (int i = 0; i < fftResultLen; ++i)
    {
        for (int k = 0; k < NFILTER; ++k)
        {
            fbank_T[i * NFILTER + k] = fbank[k * fftResultLen + i];
        }
    }

    // create buffers
    cl_mem fbank_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * fftResultLen * NFILTER, nullptr, &errNum);
    cl_mem filterBanks_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * numFrames * NFILTER, nullptr, &errNum);

    if (fbank_mem == NULL || filterBanks_mem == NULL)
    {
        std::cerr << "Failed to create fbank_mem or filterBanks_mem buffers." << std::endl;

        return 1;
    }

    // create kernel
    cl_kernel melFilBank_ker = clCreateKernel(program, "MelFilBank_test", &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create melFilBank_ker kernel." << std::endl;

        return 1;
    }

    // set kernel argument
    argCount = 0;
    errNum = clSetKernelArg(melFilBank_ker, argCount++, sizeof(int), &numFrames);
    errNum |= clSetKernelArg(melFilBank_ker, argCount++, sizeof(int), &fftResultLen);
    errNum |= clSetKernelArg(melFilBank_ker, argCount++, sizeof(int), &NFILTER);
    errNum |= clSetKernelArg(melFilBank_ker, argCount++, sizeof(cl_mem), &powFrames_mem);
    errNum |= clSetKernelArg(melFilBank_ker, argCount++, sizeof(cl_mem), &fbank_mem);
    errNum |= clSetKernelArg(melFilBank_ker, argCount++, sizeof(cl_mem), &filterBanks_mem);
    errNum |= clSetKernelArg(melFilBank_ker, argCount++, sizeof(float) * 16 * fftResultLen, nullptr);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create MelFilBank kernel argument." << std::endl;

        return 1;
    }

    errNum = clEnqueueWriteBuffer(commandQueue, fbank_mem, CL_TRUE, 0,
        sizeof(float) * fftResultLen * NFILTER, fbank_T, 0, NULL, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to write fbank_mem buffer." << std::endl;

        return 1;
    }

    // execute kernel arguments
    cl_uint nd_fb = 2;
    size_t global_fb[] = { static_cast<size_t>(ceil(numFrames / 16.0) * 16), NFILTER };
    size_t local_fb[] = { 16, NFILTER };

    //----------------------------------//
    //              MFCCs               //
    //----------------------------------//

    // create buffers
    cl_mem mfcc_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * numFrames * NUM_CEPS, nullptr, &errNum);

    if (mfcc_mem == NULL)
    {
        std::cerr << "Failed to create fbank_mem or mfcc_mem buffers." << std::endl;

        return 1;
    }

    // create kernel
    cl_kernel dct_ker = clCreateKernel(program, "DCT", &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create dct_ker kernel." << std::endl;

        return 1;
    }

    // set kernel argument
    argCount = 0;
    errNum = clSetKernelArg(dct_ker, argCount++, sizeof(int), &numFrames);
    errNum |= clSetKernelArg(dct_ker, argCount++, sizeof(int), &NFILTER);
    errNum |= clSetKernelArg(dct_ker, argCount++, sizeof(int), &NUM_CEPS);
    errNum |= clSetKernelArg(dct_ker, argCount++, sizeof(float), &CEP_LIFTER);
    errNum |= clSetKernelArg(dct_ker, argCount++, sizeof(cl_mem), &filterBanks_mem);
    errNum |= clSetKernelArg(dct_ker, argCount++, sizeof(cl_mem), &mfcc_mem);
    errNum |= clSetKernelArg(dct_ker, argCount++, sizeof(float) * 4 * NFILTER, NULL);
    errNum |= clSetKernelArg(dct_ker, argCount++, sizeof(float) * NUM_CEPS * NFILTER, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create DCT kernel argument." << std::endl;

        return 1;
    }

    // execute kernel arguments
    cl_uint nd_dct = 2;
    size_t global_dct[] = { static_cast<size_t>(ceil(numFrames / 4.0) * 4), NFILTER };
    size_t local_dct[] = { 4, NFILTER };

    //----------------------------------//
    //         Mean Normalization       //
    //----------------------------------//

    // create kernel
    cl_kernel meanNorm_ker = clCreateKernel(program, "MeanNorm", &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create meanNorm_ker kernel." << std::endl;

        return 1;
    }

    // set kernel argument
    argCount = 0;
    errNum = clSetKernelArg(meanNorm_ker, argCount++, sizeof(int), &numFrames);
    errNum |= clSetKernelArg(meanNorm_ker, argCount++, sizeof(int), &NUM_CEPS);
    errNum |= clSetKernelArg(meanNorm_ker, argCount++, sizeof(cl_mem), &mfcc_mem);
    errNum |= clSetKernelArg(meanNorm_ker, argCount++, sizeof(float) * numFrames / 2, nullptr);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to create MeanNorm kernel argument." << std::endl;

        return 1;
    }

    // execute kernel arguments
    cl_uint nd_mean = 1;
    size_t global_mean[] = { numFrames / 2, NUM_CEPS };
    size_t local_mean[] = { numFrames / 2, 1 };

    //----------------------------------//
    //           Execute Kernels        //
    //----------------------------------//

    errNum = clEnqueueWriteBuffer(commandQueue, signal_mem, CL_TRUE, 0, sizeof(short) * signalLen, signal, 0, nullptr, nullptr);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to write buffer." << std::endl;

        return 0;
    }

    cl_event pre_ev, fft_ev, mel_ev, mfcc_ev, mean_ev;

    // pre-processing
    errNum = clEnqueueNDRangeKernel(commandQueue, pre_ker, nd_pre, NULL, global_pre, local_pre, 0, nullptr, &pre_ev);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to enqueue OpenCL NDRange pre_t_ker." << std::endl;

        if (errNum == CL_INVALID_WORK_GROUP_SIZE)
        {
            std::cerr << "CL_INVALID_WORK_GROUP_SIZE" << std::endl;
        }

        return 1;
    }

    // fft
    errNum = clEnqueueNDRangeKernel(commandQueue, powFFT_ker, nd_fft, NULL, global_fft, local_fft, 0, NULL, &fft_ev);
    
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to enqueue OpenCL NDRange framing_ker." << std::endl;
    
        if (errNum == CL_INVALID_WORK_GROUP_SIZE)
        {
            std::cerr << "CL_INVALID_WORK_GROUP_SIZE" << std::endl;
        }
    
        return 1;
    }
    
    // filterbank
    errNum = clEnqueueNDRangeKernel(commandQueue, melFilBank_ker, nd_fb, NULL, global_fb, local_fb, 0, NULL, &mel_ev);
    
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to enqueue OpenCL NDRange melFilBank_ker." << std::endl;
    
        if (errNum == CL_INVALID_WORK_GROUP_SIZE)
        {
            std::cerr << "CL_INVALID_WORK_GROUP_SIZE" << std::endl;
        }
        else if (errNum == CL_INVALID_WORK_ITEM_SIZE)
        {
            std::cerr << "CL_INVALID_WORK_ITEM_SIZE" << std::endl;
        }
    
        return 1;
    }
    
    // mfcc
    errNum = clEnqueueNDRangeKernel(commandQueue, dct_ker, nd_dct, NULL, global_dct, local_dct, 0, NULL, &mfcc_ev);
    
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to enqueue OpenCL NDRange dct_ker." << std::endl;
    
        if (errNum == CL_INVALID_WORK_GROUP_SIZE)
        {
            std::cerr << "CL_INVALID_WORK_GROUP_SIZE" << std::endl;
        }
        else if (errNum == CL_INVALID_WORK_ITEM_SIZE)
        {
            std::cerr << "CL_INVALID_WORK_ITEM_SIZE" << std::endl;
        }
    
        return 1;
    }
    
    // mean norm
    errNum = clEnqueueNDRangeKernel(commandQueue, meanNorm_ker, nd_mean, NULL, global_mean, local_mean, 0, NULL, &mean_ev);
    
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to enqueue OpenCL NDRange meanNorm_ker." << std::endl;
    
        if (errNum == CL_INVALID_WORK_GROUP_SIZE)
        {
            std::cerr << "CL_INVALID_WORK_GROUP_SIZE" << std::endl;
        }
        else if (errNum == CL_INVALID_WORK_ITEM_SIZE)
        {
            std::cerr << "CL_INVALID_WORK_ITEM_SIZE" << std::endl;
        }
    
        return 1;
    }
    
    errNum = clFinish(commandQueue);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to finish commandqueue." << std::endl;
    
        if (errNum == CL_INVALID_COMMAND_QUEUE)
        {
            std::cerr << "CL_INVALID_COMMAND_QUEUE" << std::endl;
        }
    
        return 0;
    }

#ifdef PROFILING

    //----------------------------------//
    //             Profiling            //
    //----------------------------------//

    std::cout << std::endl << std::endl;

    cl_ulong ev_start_time = (cl_ulong)0;
    cl_ulong ev_end_time = (cl_ulong)0;

    // pre-processing
    errNum = clGetEventProfilingInfo(pre_ev, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &ev_start_time, NULL);
    errNum |= clGetEventProfilingInfo(pre_ev, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &ev_end_time, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to profilling." << std::endl;

        return 1;
    }

    std::cout << "Pre-Processing:\t\t" << ev_end_time - ev_start_time << std::endl;

    // fft
    errNum = clGetEventProfilingInfo(fft_ev, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &ev_start_time, NULL);
    errNum |= clGetEventProfilingInfo(fft_ev, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &ev_end_time, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to profilling." << std::endl;

        return 1;
    }

    std::cout << "FFT:\t\t\t" << ev_end_time - ev_start_time << std::endl;

    // mel_filter
    errNum = clGetEventProfilingInfo(mel_ev, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &ev_start_time, NULL);
    errNum |= clGetEventProfilingInfo(mel_ev, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &ev_end_time, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to profilling." << std::endl;

        return 1;
    }

    std::cout << "Mel-Filter:\t\t" << ev_end_time - ev_start_time << std::endl;

    // mfcc
    errNum = clGetEventProfilingInfo(mfcc_ev, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &ev_start_time, NULL);
    errNum |= clGetEventProfilingInfo(mfcc_ev, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &ev_end_time, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to profilling." << std::endl;

        return 1;
    }

    std::cout << "MFCC:\t\t\t" << ev_end_time - ev_start_time << std::endl;

    // mean normalization
    errNum = clGetEventProfilingInfo(mean_ev, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &ev_start_time, NULL);
    errNum |= clGetEventProfilingInfo(mean_ev, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &ev_end_time, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to profilling." << std::endl;

        return 1;
    }

    std::cout << "Mean Norm:\t\t" << ev_end_time - ev_start_time << std::endl;

    std::cout << std::endl << std::endl;

#endif

    return 0;
}
