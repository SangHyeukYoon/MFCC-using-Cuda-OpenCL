#include "Wav.h"

#include <iostream>

Wav::Wav(const char* filePath)
{
	fopen_s(&fp, filePath, "rb");

	fread_s(&_header_riff, sizeof(RIFF), sizeof(RIFF), 1, fp);
	fread_s(&_header_fmt, sizeof(FMT), sizeof(FMT), 1, fp);
	fread_s(&_header_data, sizeof(DATA), sizeof(DATA), 1, fp);
}

void Wav::GetData(short* audioData)
{
	fread_s(audioData, _header_data.chunkSize, _header_data.chunkSize, 1, fp);

	for (int i = 1; i < expand; ++i)
	{
		memcpy_s(&audioData[i * _header_data.chunkSize / (_header_fmt.bitPerSample / 8)], _header_data.chunkSize / (_header_fmt.bitPerSample / 8),
			audioData, _header_data.chunkSize / (_header_fmt.bitPerSample / 8));
	}
}

const unsigned int Wav::GetLen()
{
	return _header_data.chunkSize / (_header_fmt.bitPerSample / 8) * expand;
}

Wav::~Wav()
{
	fclose(fp);
}