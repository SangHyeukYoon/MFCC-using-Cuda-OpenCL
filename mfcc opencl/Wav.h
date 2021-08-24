#pragma once

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

    short* audioData;

public:
    Wav(const char* filePath);

    short* GetData();
    const unsigned int GetLen();

    ~Wav();
};
