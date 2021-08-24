#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>

constexpr unsigned int NUMTHREAD = 12;

void DoThread(const int signalLen, const int frameLength, 
    const int frameStep, const int numFrames, 
    const int threadNum, 
    float* input, float* output);

int main()
{
    const int signalLen = 66968;
    const int frameLength = 512;
    const int frameStep = 256;
    const int numFrames = 261;
    
    float* input = new float[signalLen];

    for (unsigned int i = 0; i < signalLen; ++i)
    {
        input[i] = i * 0.003;
    }

    float* output = new float[numFrames * frameLength];

    std::thread t[NUMTHREAD];

    std::chrono::system_clock::time_point StartTime = std::chrono::system_clock::now();

    for (int i = 0; i < NUMTHREAD; ++i)
    {
        t[i] = std::thread(DoThread, signalLen, frameLength, frameStep, numFrames, i, input, output);
    }

    for (int i = 0; i < NUMTHREAD; ++i)
    {
        if (t[i].joinable())
        {
            t[i].join();
        }
    }

    std::chrono::system_clock::time_point EndTime = std::chrono::system_clock::now();

    std::chrono::nanoseconds nano = EndTime - StartTime;

    std::cout << "Compute Time:\t" << nano.count() << std::endl;

    delete [] input;
    delete [] output;

    return 0;
}

void DoThread(const int signalLen, const int frameLength, 
    const int frameStep, const int numFrames, 
    const int threadNum, 
    float* input, float* output)
{
    for (int i = threadNum; i < numFrames; i += NUMTHREAD)
    {
        for (int k = 0; k < frameLength; ++k)
        {
            if (i * frameStep + k < signalLen)
            {
                output[i * frameLength + k] = input[i * frameStep + k] * 
                    (0.54 - 0.46 * cos(2 * M_PI * k / (frameLength - 1)));
            }
            else
            {
                output[i * frameLength + k] = 0;
            }
        }
    }
}
