#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <stdlib.h>
#include <string>

#include <chrono>
#include <cmath>

#include "kernel.h"
#include "Wav.h"

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

int main()
{
    cudaError_t cudaStatus;

    // load wav file
    Wav wav{ FILE_PATH };

    const int signalLen = wav.GetLen();
    const int sr = 16000;

    short* signal_h;

    cudaStatus = cudaMallocHost((void**)&signal_h, signalLen * sizeof(short));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocHost failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    wav.GetData(signal_h);

    const int frameLength = FRAME_SIZE * sr;
    const int frameStep = FRAME_STRIDE * sr;
    const int numFrames = ceil(float(signalLen - frameLength) / frameStep) + 1;

    // create events
    cudaEvent_t startEvent, stopEvent, dummyEvent;

    cudaStatus = cudaEventCreate(&startEvent);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaEventCreate: startEvent, failed!");

        return 1;
    }

    cudaStatus = cudaEventCreate(&stopEvent);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaEventCreate: stopEvent, failed!");

        return 1;
    }

    cudaStatus = cudaEventCreate(&dummyEvent);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaEventCreate: dummyEvent, failed!");

        return 1;
    }

    //----------------------------------//
    //           Pre-Processing         //
    //----------------------------------//

    short* signal_d;
    float* emphasised_d;

    cudaStatus = cudaMalloc((void**)&signal_d, signalLen * sizeof(short));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    cudaStatus = cudaMalloc((void**)&emphasised_d, signalLen * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    //----------------------------------//
    //      FFT and Power Spectrum      //
    //----------------------------------//

    float* powFrames_d;

    cudaStatus = cudaMalloc((void**)&powFrames_d, numFrames * (NFFT / 2 + 1) * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    //----------------------------------//
    //           Filter Banks           //
    //----------------------------------//

    float lowFreqMel = 0;
    float highFreqMel = 2595.0 * log10(1 + (sr / 2.0) / 700.0);
    float melPoints[NFILTER + 2];

    float melStride = (highFreqMel - lowFreqMel) / (NFILTER + 1);
    for (int i = 0; i < NFILTER + 2; ++i)
    {
        melPoints[i] = lowFreqMel + melStride * i;
    }

    float melBins[NFILTER + 2];
    for (int i = 0; i < NFILTER + 2; ++i)
    {
        melBins[i] = floor((NFFT + 1) * (700.0 * (pow(10, (melPoints[i] / 2595.0)) - 1)) / float(sr));
    }

    const int fftResultLen = NFFT / 2 + 1;

    float* fbank = new float[NFILTER * fftResultLen];
    memset(fbank, 0, sizeof(float) * (NFILTER * fftResultLen));

    int nonZeroNum = 0;

    float f_m_minus, f_m, f_m_plus;
    for (int m = 1; m <= NFILTER; ++m)
    {
        f_m_minus = melBins[m - 1];
        f_m = melBins[m];
        f_m_plus = melBins[m + 1];

        for (int k = f_m_minus; k < f_m; ++k)
        {
            fbank[(m - 1) * fftResultLen + k] = (k - f_m_minus) / (f_m - f_m_minus);

            if (fbank[(m - 1) * fftResultLen + k] != 0)
            {
                ++nonZeroNum;
            }
        }

        for (int k = f_m; k < f_m_plus; ++k)
        {
            fbank[(m - 1) * fftResultLen + k] = (f_m_plus - k) / (f_m_plus - f_m);
            
            if (fbank[(m - 1) * fftResultLen + k] != 0)
            {
                ++nonZeroNum;
            }
        }
    }

    float* fbanks_val_h;
    int* fbanks_col_h;
    int* fbanks_row_h;

    cudaMallocHost((void**)&fbanks_val_h, nonZeroNum * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocHost failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    cudaMallocHost((void**)&fbanks_col_h, (NFILTER + 2) * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocHost failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    cudaMallocHost((void**)&fbanks_row_h, nonZeroNum * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocHost failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    fbanks_col_h[0] = 0;
    nonZeroNum = 0;

    for (int i = 0; i < NFILTER; ++i)
    {
        for (int k = 0; k < fftResultLen; ++k)
        {
            if (fbank[i * fftResultLen + k] != 0)
            {
                fbanks_val_h[nonZeroNum] = fbank[i * fftResultLen + k];
                fbanks_row_h[nonZeroNum++] = k;
            }
        }

        fbanks_col_h[i + 1] = nonZeroNum;
    }

    fbanks_col_h[NFILTER + 1] = nonZeroNum;

    float* fbanks_val_d;
    int* fbanks_col_d;
    int* fbanks_row_d;

    float* filterBanks_d;   // filter banks output

    cudaMalloc((void**)&fbanks_val_d, nonZeroNum * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocHost failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    cudaMalloc((void**)&fbanks_col_d, (NFILTER + 2) * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocHost failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    cudaMalloc((void**)&fbanks_row_d, nonZeroNum * sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocHost failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    cudaStatus = cudaMalloc((void**)&filterBanks_d, numFrames * NFILTER * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    cudaStatus = cudaMemcpy(fbanks_val_d, fbanks_val_h, nonZeroNum * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed! signal_d: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    cudaStatus = cudaMemcpy(fbanks_col_d, fbanks_col_h, (NFILTER + 2) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed! signal_d: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    cudaStatus = cudaMemcpy(fbanks_row_d, fbanks_row_h, nonZeroNum * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed! signal_d: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    //----------------------------------//
    //              MFCCs               //
    //----------------------------------//

    float* mfcc_d;
    cudaStatus = cudaMalloc((void**)&mfcc_d, numFrames * NUM_CEPS * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));
    
        return 1;
    }
    
    //----------------------------------//
    //         Mean Normalization       //
    //----------------------------------//

    float* mfcc_h;

    cudaStatus = cudaMallocHost((void**)&mfcc_h, numFrames * NUM_CEPS * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMallocHost failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    int p2 = pow(2, ceil(log(numFrames / 2) / log(2)));

    float* meanNorm_d;
    cudaStatus = cudaMalloc((void**)&meanNorm_d, p2 * NUM_CEPS * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    //----------------------------------//
    //          Execute Kernels         //
    //----------------------------------//

    float ms = 0.0;

    cudaStatus = cudaEventRecord(startEvent);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaEventRecord: startEvent, failed!");

        return 1;
    }

    cudaStatus = cudaMemcpy(signal_d, signal_h, signalLen * sizeof(short), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed! signal_d: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    PreEmphasis <<< ceil(double(signalLen) / 1024.0), 1024 >>> (PRE_EMPHASIS, signalLen, signal_d, emphasised_d);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "PreEmphasis launch failed: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    PowerFFT <<< numFrames, NFFT / 2, 512 * 3 * sizeof(float2) >>> (BIT_LENGTH, frameLength, frameStep, signalLen, emphasised_d, powFrames_d);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "PowerFFT launch failed: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    MelFilterBank_Sparse <<< numFrames, NFILTER >>> (numFrames, fftResultLen, NFILTER, powFrames_d, filterBanks_d, fbanks_val_d, fbanks_col_d, fbanks_row_d);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "MelFilterBank_Sparse launch failed: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    DCT <<< numFrames, NFILTER >>> (numFrames, NFILTER, NUM_CEPS, CEP_LIFTER, filterBanks_d, mfcc_d);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "DCT launch failed: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    if (p2 > 1024)
    {
        MeanNorm_global << < NUM_CEPS, p2 / 4 >> > (numFrames, NUM_CEPS, mfcc_d, meanNorm_d);
        MeanNorm_global << < NUM_CEPS, p2 / 4 >> > (numFrames, NUM_CEPS, mfcc_d, meanNorm_d);
        MeanNorm_global << < NUM_CEPS, p2 / 4 >> > (numFrames, NUM_CEPS, mfcc_d, meanNorm_d);
        MeanNorm_global << < NUM_CEPS, p2 / 4 >> > (numFrames, NUM_CEPS, mfcc_d, meanNorm_d);
    }
    else if (p2 > 512)
    {
        MeanNorm_global <<< NUM_CEPS, p2 >>> (numFrames, NUM_CEPS, mfcc_d, meanNorm_d);
    }
    else
    {
        MeanNorm <<< NUM_CEPS, p2, p2 * sizeof(float) >>> (numFrames, NUM_CEPS, mfcc_d);
    }
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "MeanNorm launch failed: %s\n", cudaGetErrorString(cudaStatus));
    
        return 1;
    }

    cudaStatus = cudaMemcpy(mfcc_h, mfcc_d, numFrames * NUM_CEPS * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed! signal_d: %s\n", cudaGetErrorString(cudaStatus));

        return 1;
    }

    // stop to record event. 
    cudaStatus = cudaEventRecord(stopEvent);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaEventRecord: stopEvent, failed!");

        return 1;
    }

    cudaStatus = cudaEventSynchronize(stopEvent);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaEventSynchronize: stopEvent, failed!");

        return 1;
    }

    cudaStatus = cudaEventElapsedTime(&ms, startEvent, stopEvent);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaEventElapsedTime failed!");

        return 1;
    }

    printf("Serial Time:\t%f\n", ms * 1e+6);

    PrintArray(NUM_CEPS, mfcc_h + (numFrames - 1) * NUM_CEPS);

    //----------------------------------//
    //              Cleaning            //
    //----------------------------------//

    // free host memory
    cudaFreeHost(signal_h);

    cudaFreeHost(fbanks_val_h);
    cudaFreeHost(fbanks_col_h);
    cudaFreeHost(fbanks_row_h);

    cudaFreeHost(mfcc_h);

    // free device memory
    cudaFree(signal_d);
    cudaFree(emphasised_d);

    cudaFree(powFrames_d);

    cudaFree(filterBanks_d);

    cudaFree(fbanks_val_d);
    cudaFree(fbanks_col_d);
    cudaFree(fbanks_row_d);

    cudaFree(meanNorm_d);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");

        return 1;
    }

    return 0;
}
