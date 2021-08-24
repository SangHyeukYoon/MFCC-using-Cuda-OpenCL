#pragma once

#include <iostream>

class Wav
{
private:
    struct RIFF
    {
        unsigned char chunkID[4];
        unsigned int chunkSize;
        unsigned char format[4];
    } _header_riff;

    struct FMT
    {
        unsigned char chunkID[4];
        unsigned int chunkSize;
        unsigned short audioFormat;
        unsigned short numChannels;
        unsigned int sampleRate;
        unsigned int avgByteRate;
        unsigned short blockAlign;
        unsigned short bitPerSample;
    } _header_fmt;

    struct DATA
    {
        unsigned char chunkID[4];
        unsigned int chunkSize;
    } _header_data;

    FILE* fp;

    const int expand = 8;

public:
    Wav(const char* filePath);

    void GetData(short* audioData);
    const unsigned int GetLen();

    ~Wav();
};
