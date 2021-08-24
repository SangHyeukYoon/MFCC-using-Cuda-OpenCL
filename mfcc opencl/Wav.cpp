#include "Wav.h"

#include <iostream>

Wav::Wav(const char* filePath)
{
	FILE* fp;
	fopen_s(&fp, filePath, "rb");

	fread_s(&_header_riff, sizeof(RIFF), sizeof(RIFF), 1, fp);
	fread_s(&_header_fmt, sizeof(FMT), sizeof(FMT), 1, fp);
	fread_s(&_header_data, sizeof(DATA), sizeof(DATA), 1, fp);

	int size = _header_data.chunkSize / (_header_fmt.bitPerSample / 8);

	audioData = new short[size];

	fread_s(audioData, _header_data.chunkSize, _header_data.chunkSize, 1, fp);

	fclose(fp);
}

short* Wav::GetData()
{
    return audioData;
}

const unsigned int Wav::GetLen()
{
    return _header_data.chunkSize / (_header_fmt.bitPerSample / 8);
}

Wav::~Wav()
{
    if (audioData != nullptr)
    {
        delete[] audioData;
    }
}