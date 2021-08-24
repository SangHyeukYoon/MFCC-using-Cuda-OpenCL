#include "kernel.h"
#include <stdio.h>

#define M_PI CUDART_PI_F

__global__
void PreEmphasis(const float PRE_EMPHASIS, const int signalLen,
	short* signal, float* emphasised)
{
	const int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id > signalLen)
	{
		return;
	}

	if (id == 0)
	{
		emphasised[0] = float(signal[0]);
	}
	else
	{
		emphasised[id] = signal[id] - PRE_EMPHASIS * signal[id - 1];
	}
}

__global__
void PreProcessing(const float PRE_EMPHASIS, const int frameLength, const int frameStep, const int signalLen,
	short* signal, float* frames)
{
	// thread and block id
	const int id = threadIdx.x;
	const int frameNum = blockIdx.x;

	// declare shared memory. 
	__shared__ short signal_s[514];
	__shared__ float emphasis_s[512];

	// load signal to shared memory. 
	if (id == 0)
	{
		if (frameNum == 0)
		{
			signal_s[0] = 0;
		}
		else
		{
			signal_s[0] = signal[frameStep * frameNum - 1];
		}
	}

	if (frameStep * frameNum + id > signalLen)
	{
		signal_s[id + 1] = 0;
	}
	else
	{
		signal_s[id + 1] = signal[frameStep * frameNum + id];
	}

	__syncthreads();

	if (frameStep * frameNum + id >= signalLen)
	{
		emphasis_s[id] = 0;
	}
	else
	{
		// pre-emphasis
		emphasis_s[id] = signal_s[id + 1] - PRE_EMPHASIS * signal_s[id];
	}

	__syncthreads();

	// framing
	//frames[frameLength * frameNum + id] = emphasis_s[id] *
	//	(0.54 - 0.46 * cos(2 * M_PI * id / (frameLength - 1)));

	frames[frameLength * frameNum + id] = emphasis_s[id] * HammingFilter[id];
}

__device__
float2 CAdd(const float2 a, const float2 b)
{
	float2 result;

	result.x = a.x + b.x;
	result.y = a.y + b.y;

	return result;
}

__device__
float2 CSub(const float2 a, const float2 b)
{
	float2 result;

	result.x = a.x - b.x;
	result.y = a.y - b.y;

	return result;
}

__device__
float2 CMul(const float2 a, const float2 b)
{
	float2 result;

	result.x = (a.x * b.x) - (a.y * b.y);
	result.y = (a.x * b.y) + (a.y * b.x);

	return result;
}

__global__
void PowerFFT(const int bitLength, const int frameLength, const int frameStep, const int signalLen,
	float* input, float* output)
{
	const int id = threadIdx.x;
	const int frameNum = blockIdx.x;

	const int N = 1 << bitLength;
	int l = 1 << (bitLength - 1);
	int m = 1;

	int j, k;

	float x;

	extern __shared__ float2 s[];

	__shared__ float2* input_s;
	__shared__ float2* output_s;
	__shared__ float2* twiddle_s;

	input_s = s;
	output_s = &input_s[512];
	twiddle_s = &output_s[512];

	__shared__ float2* tmp_s;

	// copy global data to shared memory. 
	if (frameNum * frameStep + id >= signalLen)
	{
		input_s[id].x = 0;
	}
	else
	{
		input_s[id].x = input[frameNum * frameStep + id] *
			(0.54 - 0.46 * cos(2 * M_PI * id / (frameLength - 1)));
	}

	input_s[id].y = 0;

	if (frameNum * frameStep + (id + l) >= signalLen)
	{
		input_s[id + l].x = 0;
	}
	else
	{
		input_s[id + l].x = input[frameNum * frameStep + (id + l)] *
			(0.54 - 0.46 * cos(2 * M_PI * (id + l) / (frameLength - 1)));
	}

	input_s[id + l].y = 0;

	// calculate twiddle factor
	twiddle_s[0] = make_float2(1.0f, 0.0f);

	int lo = id + 1;	// location to compute
	int log = 0;		// integer value of log

	while ((lo >>= 1) > 0)
	{
		++log;
	}

	lo = id + 1;

	int j_t = (lo - (1 << log)) * 2;      // j value of twiddle
	int l_t = 1 << (log + 1);             // l value of twiddle

	x = -M_PI * j_t / l_t;
	twiddle_s[l_t - 1 + j_t] = make_float2(cos(x), sin(x));

	++j_t;

	x = -M_PI * j_t / l_t;
	twiddle_s[l_t - 1 + j_t] = make_float2(cos(x), sin(x));

	__syncthreads();

	// fft
	for (l = 1 << (bitLength - 1); l >= 1; l >>= 1, m <<= 1)
	{
		j = id / m;
		k = id % m;

		output_s[k + 2 * j * m] = CAdd(input_s[k + j * m], input_s[k + j * m + l * m]);
		output_s[k + 2 * j * m + m] = CMul(twiddle_s[l - 1 + j], CSub(input_s[k + j * m], input_s[k + j * m + l * m]));

		__syncthreads();

		tmp_s = input_s;
		input_s = output_s;
		output_s = tmp_s;
	}

	output[frameNum * ((N / 2) + 1) + id] = (input_s[id].x * input_s[id].x + input_s[id].y * input_s[id].y) / N;

	if (id == 0)
	{
		int i = N >> 1;

		output[frameNum * (i + 1) + i] = (input_s[i].x * input_s[i].x + input_s[i].y * input_s[i].y) / N;
	}
}

__global__
void MelFilterBank(const int numFrames, const int fftLen, const int nFilter, const int threadSize,
	float* powFrames, float* fbank, float* filterBanks)
{
	const int id_y = threadIdx.x;
	const int id_x = threadIdx.y;

	const int gridIdx_y = blockIdx.x;
	const int gridIdx_x = blockIdx.y;

	__shared__ float powFrames_s[20][20];
	__shared__ float fbank_s[20][20];

	float value = 1.19209e-07;
	int y = (threadSize * gridIdx_y + id_y);

	const int repeat = (int)(fftLen / threadSize) + 1;

	for (int count = 0; count < repeat; ++count)
	{
		// load powFrames
		if (y < numFrames && count * threadSize + id_x < fftLen)
		{
			powFrames_s[id_y][id_x] = powFrames[y * fftLen + count * threadSize + id_x];
		}
		else
		{
			powFrames_s[id_y][id_x] = 0;
		}

		// load filterBanks
		if (count * threadSize + id_y < fftLen)
		{
			fbank_s[id_y][id_x] = fbank[(count * threadSize + id_y) * nFilter + gridIdx_x * threadSize + id_x];
		}
		else
		{
			fbank_s[id_y][id_x] = 0;
		}

		__syncthreads();

		for (int k = 0; k < threadSize; ++k)
		{
			value += powFrames_s[id_y][k] * fbank_s[k][id_x];
		}

		__syncthreads();
	}

	if (y < numFrames)
	{
		filterBanks[y * nFilter + gridIdx_x * threadSize + id_x] = 20 * log10(value);
	}
}

__global__
void MelFilterBank_Sparse(const int numFrames, const int fftLen, const int nFilter,
	float* powFrames, float* filterBanks,
	float* fbanks_val, int* fbanks_col, int* fbanks_row)
{
	const int id = threadIdx.x;
	const int blockId = blockIdx.x;

	int startNum = fbanks_col[id];
	int endNum = fbanks_col[id + 1];

	float value = 1.19209e-07;

	for (int k = startNum; k < endNum; ++k)
	{
		value += fbanks_val[k] * powFrames[blockId * fftLen + fbanks_row[k]];
	}

	filterBanks[blockId * nFilter + id] = 20 * log10(value);
}

__global__
void DCT(const int numFrames, const int nFilter, const int numCeps, const float cepLifter,
	float* filterBanks, float* mfcc)
{
	const int id = threadIdx.x;
	const int blockId = blockIdx.x;

	__shared__ float filterBanks_s[40];

	filterBanks_s[id] = filterBanks[nFilter * blockId + id];

	__syncthreads();

	if (id > 0 && id < numCeps + 1)
	{
		float value = 0;

		for (int k = 0; k < nFilter; ++k)
		{
			value += filterBanks_s[k] * cos(M_PI * id * (2 * k + 1) / (2 * nFilter));
		}

		float lift = 1 + (cepLifter / 2) * sin(M_PI * (id - 1) / cepLifter);

		mfcc[blockId * numCeps + id - 1] = 2 * value / sqrt(2.0 * nFilter) * lift;
	}
}

__global__
void MeanNorm(const int numFrames, const int numCeps,
	float* mfcc)
{
	const int id = threadIdx.x;
	const int blockId = blockIdx.x;
	const int blockSize = blockDim.x;

	const int halfSize = numFrames / 2;

	extern __shared__ float x[];

	__shared__ float* scratch;
	scratch = x;

	if (id < halfSize)
	{
		scratch[id] = mfcc[id * numCeps + blockId] + mfcc[(id + halfSize) * numCeps + blockId];
	}
	else
	{
		scratch[id] = 0;
	}

	if (numFrames % 2 == 1)
	{
		if (id == 0)
		{
			scratch[id] += mfcc[(numFrames - 1) * numCeps + blockId];
		}
	}

	__syncthreads();

	for (unsigned int stride = blockSize >> 1; id < stride; stride >>= 1)
	{
		scratch[id] += scratch[id + stride];

		__syncthreads();
	}

	float mean = scratch[0] / float(numFrames);

	if (id < halfSize)
	{
		mfcc[id * numCeps + blockId] /= mean;
		mfcc[(id + halfSize) * numCeps + blockId] /= mean;

		if (id == 0)
		{
			mfcc[(numFrames - 1) * numCeps + blockId] /= mean;
		}
	}
}

__global__
void MeanNorm_global(const int numFrames, const int numCeps,
	float* mfcc, float* meanNorm)
{
	const int id = threadIdx.x;
	const int blockId = blockIdx.x;
	const int blockSize = blockDim.x;

	const int halfSize = numFrames / 2;

	if (id < halfSize)
	{
		meanNorm[id * numCeps + blockId] = mfcc[id * numCeps + blockId] + mfcc[(id + halfSize) * numCeps + blockId];
	}
	else
	{
		meanNorm[id * numCeps + blockId] = 0;
	}

	if (numFrames % 2 == 1)
	{
		if (id == 0)
		{
			meanNorm[blockId] += mfcc[(numFrames - 1) * numCeps + blockId];
		}
	}

	__syncthreads();

	for (int stride = blockSize >> 1; id < stride; stride >>= 1)
	{
		meanNorm[id * numCeps + blockId] += meanNorm[(id + stride) * numCeps + blockId];

		__syncthreads();
	}

	float mean = meanNorm[blockId] / float(numFrames) + 1.19209e-07;

	if (id < numFrames)
	{
		mfcc[id * numCeps + blockId] -= mean;
	}

	//if (id < halfSize)
	//{
	//	mfcc[id * numCeps + blockId] -= mean;
	//	mfcc[(id + halfSize) * numCeps + blockId] -= mean;
	//
	//	if (id == 0)
	//	{
	//		mfcc[(numFrames - 1) * numCeps + blockId] -= mean;
	//	}
	//}
}
