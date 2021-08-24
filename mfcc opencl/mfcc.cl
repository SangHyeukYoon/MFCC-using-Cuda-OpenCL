__constant float HammingFilter[] = 
{ 
    0.08, 0.0800348, 0.0801391, 0.0803129, 0.0805563, 0.0808691, 0.0812513, 0.0817028,
    0.0822237, 0.0828138, 0.083473, 0.0842012, 0.0849983, 0.0858642, 0.0867987, 0.0878018,
    0.0888733, 0.0900129, 0.0912206, 0.0924962, 0.0938394, 0.09525, 0.0967279, 0.0982728,
    0.0998845, 0.101563, 0.103307, 0.105118, 0.106994, 0.108936, 0.110943, 0.113014,
    0.115151, 0.117351, 0.119616, 0.121944, 0.124335, 0.126789, 0.129306, 0.131884,
    0.134525, 0.137226, 0.139989, 0.142812, 0.145695, 0.148638, 0.151639, 0.1547,
    0.157819, 0.160995, 0.164229, 0.16752, 0.170867, 0.17427, 0.177728, 0.181241,
    0.184808, 0.188429, 0.192103, 0.195829, 0.199608, 0.203438, 0.207319, 0.21125,
    0.215231, 0.219261, 0.22334, 0.227466, 0.23164, 0.23586, 0.240126, 0.244438,
    0.248794, 0.253195, 0.257638, 0.262125, 0.266653, 0.271223, 0.275833, 0.280483,
    0.285173, 0.289901, 0.294667, 0.29947, 0.304309, 0.309184, 0.314094, 0.319038,
    0.324015, 0.329025, 0.334067, 0.33914, 0.344244, 0.349377, 0.354539, 0.359729,
    0.364946, 0.37019, 0.375459, 0.380753, 0.386071, 0.391413, 0.396777, 0.402162,
    0.407569, 0.412995, 0.418441, 0.423905, 0.429387, 0.434885, 0.440399, 0.445929,
    0.451472, 0.457029, 0.462599, 0.46818, 0.473772, 0.479374, 0.484985, 0.490605,
    0.496232, 0.501865, 0.507505, 0.513149, 0.518797, 0.524449, 0.530103, 0.535758,
    0.541414, 0.54707, 0.552725, 0.558377, 0.564027, 0.569674, 0.575316, 0.580952,
    0.586583, 0.592206, 0.597822, 0.603428, 0.609025, 0.614612, 0.620188, 0.625751,
    0.631301, 0.636838, 0.64236, 0.647866, 0.653356, 0.658829, 0.664284, 0.66972,
    0.675137, 0.680533, 0.685908, 0.691261, 0.696591, 0.701897, 0.707179, 0.712436,
    0.717666, 0.72287, 0.728046, 0.733193, 0.738312, 0.7434, 0.748458, 0.753484,
    0.758478, 0.763438, 0.768365, 0.773258, 0.778115, 0.782936, 0.787721, 0.792468,
    0.797177, 0.801847, 0.806477, 0.811067, 0.815616, 0.820124, 0.824589, 0.829011,
    0.833389, 0.837723, 0.842012, 0.846256, 0.850453, 0.854603, 0.858706, 0.86276,
    0.866765, 0.870722, 0.874628, 0.878483, 0.882288, 0.88604, 0.889741, 0.893388,
    0.896982, 0.900522, 0.904008, 0.907439, 0.910814, 0.914132, 0.917395, 0.9206,
    0.923748, 0.926838, 0.929869, 0.932841, 0.935754, 0.938607, 0.9414, 0.944132,
    0.946803, 0.949413, 0.95196, 0.954446, 0.956868, 0.959228, 0.961524, 0.963757,
    0.965925, 0.968029, 0.970069, 0.972043, 0.973952, 0.975796, 0.977573, 0.979285,
    0.98093, 0.982508, 0.984019, 0.985464, 0.986841, 0.98815, 0.989392, 0.990565,
    0.991671, 0.992708, 0.993677, 0.994577, 0.995409, 0.996172, 0.996865, 0.99749,
    0.998045, 0.998532, 0.998949, 0.999296, 0.999574, 0.999783, 0.999922, 0.999991,
    0.999991, 0.999922, 0.999783, 0.999574, 0.999296, 0.998949, 0.998532, 0.998045,
    0.99749, 0.996865, 0.996172, 0.995409, 0.994577, 0.993677, 0.992708, 0.991671,
    0.990565, 0.989392, 0.98815, 0.986841, 0.985464, 0.984019, 0.982508, 0.98093,
    0.979285, 0.977573, 0.975796, 0.973952, 0.972043, 0.970069, 0.968029, 0.965925,
    0.963757, 0.961524, 0.959228, 0.956868, 0.954446, 0.95196, 0.949413, 0.946803,
    0.944132, 0.9414, 0.938607, 0.935754, 0.932841, 0.929869, 0.926838, 0.923748,
    0.9206, 0.917395, 0.914132, 0.910814, 0.907439, 0.904008, 0.900522, 0.896982,
    0.893388, 0.889741, 0.88604, 0.882288, 0.878483, 0.874628, 0.870722, 0.866765,
    0.86276, 0.858706, 0.854603, 0.850453, 0.846256, 0.842012, 0.837723, 0.833389,
    0.829011, 0.824589, 0.820124, 0.815616, 0.811067, 0.806477, 0.801847, 0.797177,
    0.792468, 0.787721, 0.782936, 0.778115, 0.773258, 0.768365, 0.763438, 0.758478,
    0.753484, 0.748458, 0.7434, 0.738312, 0.733193, 0.728046, 0.72287, 0.717666,
    0.712436, 0.707179, 0.701897, 0.696591, 0.691261, 0.685908, 0.680533, 0.675137,
    0.66972, 0.664284, 0.658829, 0.653356, 0.647866, 0.64236, 0.636838, 0.631301,
    0.625751, 0.620188, 0.614612, 0.609025, 0.603428, 0.597822, 0.592206, 0.586583,
    0.580952, 0.575316, 0.569674, 0.564027, 0.558377, 0.552725, 0.54707, 0.541414,
    0.535758, 0.530103, 0.524449, 0.518797, 0.513149, 0.507505, 0.501865, 0.496232,
    0.490605, 0.484985, 0.479374, 0.473772, 0.46818, 0.462599, 0.457029, 0.451472,
    0.445929, 0.440399, 0.434885, 0.429387, 0.423905, 0.418441, 0.412995, 0.407569,
    0.402162, 0.396777, 0.391413, 0.386071, 0.380753, 0.375459, 0.37019, 0.364946,
    0.359729, 0.354539, 0.349377, 0.344244, 0.33914, 0.334067, 0.329025, 0.324015,
    0.319038, 0.314094, 0.309184, 0.304309, 0.29947, 0.294667, 0.289901, 0.285173,
    0.280483, 0.275833, 0.271223, 0.266653, 0.262125, 0.257638, 0.253195, 0.248794,
    0.244438, 0.240126, 0.23586, 0.23164, 0.227466, 0.22334, 0.219261, 0.215231,
    0.21125, 0.207319, 0.203438, 0.199608, 0.195829, 0.192103, 0.188429, 0.184808,
    0.181241, 0.177728, 0.17427, 0.170867, 0.16752, 0.164229, 0.160995, 0.157819,
    0.1547, 0.151639, 0.148638, 0.145695, 0.142812, 0.139989, 0.137226, 0.134525,
    0.131884, 0.129306, 0.126789, 0.124335, 0.121944, 0.119616, 0.117351, 0.115151,
    0.113014, 0.110943, 0.108936, 0.106994, 0.105118, 0.103307, 0.101563, 0.0998845,
    0.0982728, 0.0967279, 0.09525, 0.0938394, 0.0924962, 0.0912206, 0.0900129, 0.0888733,
    0.0878018, 0.0867987, 0.0858642, 0.0849983, 0.0842012, 0.083473, 0.0828138, 0.0822237,
    0.0817028, 0.0812513, 0.0808691, 0.0805563, 0.0803129, 0.0801391, 0.0800348, 0.08
};

__constant float dcF[] =
{
    0.999229, 0.993068, 0.980785, 0.962455, 0.938191, 0.908143, 0.872496, 0.83147,
    0.785317, 0.734322, 0.678801, 0.619094, 0.55557, 0.488621, 0.41866, 0.346117,
    0.27144, 0.19509, 0.117537, 0.0392598, -0.0392598, -0.117537, -0.19509, -0.27144,
    -0.346117, -0.41866, -0.488621, -0.55557, -0.619094, -0.678801, -0.734322, -0.785317,
    -0.83147, -0.872496, -0.908143, -0.938191, -0.962455, -0.980785, -0.993068, -0.999229,
    0.996917, 0.97237, 0.92388, 0.85264, 0.760406, 0.649448, 0.522499, 0.382683,
    0.233445, 0.0784591, -0.0784591, -0.233445, -0.382683, -0.522499, -0.649448, -0.760406,
    -0.85264, -0.92388, -0.97237, -0.996917, -0.996917, -0.97237, -0.92388, -0.85264,
    -0.760406, -0.649448, -0.522499, -0.382683, -0.233445, -0.0784591, 0.0784591, 0.233445,
    0.382683, 0.522499, 0.649448, 0.760406, 0.85264, 0.92388, 0.97237, 0.996917,
    0.993068, 0.938191, 0.83147, 0.678801, 0.488621, 0.27144, 0.0392598, -0.19509,
    -0.41866, -0.619094, -0.785317, -0.908143, -0.980785, -0.999229, -0.962455, -0.872496,
    -0.734322, -0.55557, -0.346117, -0.117537, 0.117537, 0.346117, 0.55557, 0.734322,
    0.872496, 0.962455, 0.999229, 0.980785, 0.908143, 0.785317, 0.619094, 0.41866,
    0.19509, -0.0392598, -0.27144, -0.488621, -0.678801, -0.83147, -0.938191, -0.993068,
    0.987688, 0.891007, 0.707107, 0.45399, 0.156434, -0.156434, -0.45399, -0.707107,
    -0.891007, -0.987688, -0.987688, -0.891007, -0.707107, -0.45399, -0.156434, 0.156434,
    0.45399, 0.707107, 0.891007, 0.987688, 0.987688, 0.891007, 0.707107, 0.45399,
    0.156434, -0.156434, -0.45399, -0.707107, -0.891007, -0.987688, -0.987688, -0.891007,
    -0.707107, -0.45399, -0.156434, 0.156434, 0.45399, 0.707107, 0.891007, 0.987688,
    0.980785, 0.83147, 0.55557, 0.19509, -0.19509, -0.55557, -0.83147, -0.980785,
    -0.980785, -0.83147, -0.55557, -0.19509, 0.19509, 0.55557, 0.83147, 0.980785,
    0.980785, 0.83147, 0.55557, 0.19509, -0.19509, -0.55557, -0.83147, -0.980785,
    -0.980785, -0.83147, -0.55557, -0.19509, 0.19509, 0.55557, 0.83147, 0.980785,
    0.980785, 0.83147, 0.55557, 0.19509, -0.19509, -0.55557, -0.83147, -0.980785,
    0.97237, 0.760406, 0.382683, -0.0784591, -0.522499, -0.85264, -0.996917, -0.92388,
    -0.649448, -0.233445, 0.233445, 0.649448, 0.92388, 0.996917, 0.85264, 0.522499,
    0.0784591, -0.382683, -0.760406, -0.97237, -0.97237, -0.760406, -0.382683, 0.0784591,
    0.522499, 0.85264, 0.996917, 0.92388, 0.649448, 0.233445, -0.233445, -0.649448,
    -0.92388, -0.996917, -0.85264, -0.522499, -0.0784591, 0.382683, 0.760406, 0.97237,
    0.962455, 0.678801, 0.19509, -0.346117, -0.785317, -0.993068, -0.908143, -0.55557,
    -0.0392598, 0.488621, 0.872496, 0.999229, 0.83147, 0.41866, -0.117537, -0.619094,
    -0.938191, -0.980785, -0.734322, -0.27144, 0.27144, 0.734322, 0.980785, 0.938191,
    0.619094, 0.117537, -0.41866, -0.83147, -0.999229, -0.872496, -0.488621, 0.0392598,
    0.55557, 0.908143, 0.993068, 0.785317, 0.346117, -0.19509, -0.678801, -0.962455,
    0.951057, 0.587785, 6.12323e-17, -0.587785, -0.951057, -0.951057, -0.587785, -1.83697e-16,
    0.587785, 0.951057, 0.951057, 0.587785, 3.06162e-16, -0.587785, -0.951057, -0.951057,
    -0.587785, -4.28626e-16, 0.587785, 0.951057, 0.951057, 0.587785, 5.51091e-16, -0.587785,
    -0.951057, -0.951057, -0.587785, 1.1028e-15, 0.587785, 0.951057, 0.951057, 0.587785,
    2.57238e-15, -0.587785, -0.951057, -0.951057, -0.587785, -2.69484e-15, 0.587785, 0.951057,
    0.938191, 0.488621, -0.19509, -0.785317, -0.999229, -0.734322, -0.117537, 0.55557,
    0.962455, 0.908143, 0.41866, -0.27144, -0.83147, -0.993068, -0.678801, -0.0392598,
    0.619094, 0.980785, 0.872496, 0.346117, -0.346117, -0.872496, -0.980785, -0.619094,
    0.0392598, 0.678801, 0.993068, 0.83147, 0.27144, -0.41866, -0.908143, -0.962455,
    -0.55557, 0.117537, 0.734322, 0.999229, 0.785317, 0.19509, -0.488621, -0.938191,
    0.92388, 0.382683, -0.382683, -0.92388, -0.92388, -0.382683, 0.382683, 0.92388,
    0.92388, 0.382683, -0.382683, -0.92388, -0.92388, -0.382683, 0.382683, 0.92388,
    0.92388, 0.382683, -0.382683, -0.92388, -0.92388, -0.382683, 0.382683, 0.92388,
    0.92388, 0.382683, -0.382683, -0.92388, -0.92388, -0.382683, 0.382683, 0.92388,
    0.92388, 0.382683, -0.382683, -0.92388, -0.92388, -0.382683, 0.382683, 0.92388,
    0.908143, 0.27144, -0.55557, -0.993068, -0.734322, 0.0392598, 0.785317, 0.980785,
    0.488621, -0.346117, -0.938191, -0.872496, -0.19509, 0.619094, 0.999229, 0.678801,
    -0.117537, -0.83147, -0.962455, -0.41866, 0.41866, 0.962455, 0.83147, 0.117537,
    -0.678801, -0.999229, -0.619094, 0.19509, 0.872496, 0.938191, 0.346117, -0.488621,
    -0.980785, -0.785317, -0.0392598, 0.734322, 0.993068, 0.55557, -0.27144, -0.908143,
    0.891007, 0.156434, -0.707107, -0.987688, -0.45399, 0.45399, 0.987688, 0.707107,
    -0.156434, -0.891007, -0.891007, -0.156434, 0.707107, 0.987688, 0.45399, -0.45399,
    -0.987688, -0.707107, 0.156434, 0.891007, 0.891007, 0.156434, -0.707107, -0.987688,
    -0.45399, 0.45399, 0.987688, 0.707107, -0.156434, -0.891007, -0.891007, -0.156434,
    0.707107, 0.987688, 0.45399, -0.45399, -0.987688, -0.707107, 0.156434, 0.891007
};

// Pre-Emphasis kernel
__kernel void PreEmphasis(float PRE_EMPHASIS, int signalLen,
    __global short* signal, __global float* output)
{
    int i = get_global_id(0);

    if (i == 0)
    {
        output[0] = signal[0];
    }
    else if (i < signalLen)
    {
        output[i] = signal[i] - PRE_EMPHASIS * signal[i - 1];
    }
}

// framing kernel
// framing + windowing
__kernel void Framing(int frameLength, int frameStep, int signalLen,
    __global float* input, __global float* output)
{
    int id_y = get_global_id(0);
    int id_x = get_global_id(1);

    if (frameStep * id_y + id_x < signalLen)
    {
        output[frameLength * id_y + id_x] = input[frameStep * id_y + id_x] * HammingFilter[id_x];
        //output[frameLength * id_y + id_x] = input[frameStep * id_y + id_x] * 
        //    (0.54 - 0.46 * cos(2 * M_PI * id_x / (frameLength - 1)))
    }
    else
    {
        output[frameLength * id_y + id_x] = 0;
    }
}

// pre processing
// pre-emphasis + framing
__kernel void PreProcessing(float PRE_EMPHASIS, int frameLength, int frameStep, int signalLen,
    __global short* signal, __global float* output,
    __local short* signal_l, __local float* emphasis_l)
{
    // basic constants
    const int id = get_global_id(1);
    const int groupID = get_global_id(0);

    // load signal to local memory. 
    if (id == 0)
    {
        if (groupID == 0)
        {
            signal_l[0] = 0;
        }
        else
        {
            signal_l[0] = signal[frameStep * groupID - 1];
        }
    }

    if (frameStep * groupID + id > signalLen)
    {
        signal_l[id + 1] = 0;
    }
    else
    {
        signal_l[id + 1] = signal[frameStep * groupID + id];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // pre-emphasis
    emphasis_l[id] = signal_l[id + 1] - PRE_EMPHASIS * signal_l[id];

    barrier(CLK_LOCAL_MEM_FENCE);

    // framing
    output[frameLength * groupID + id] = emphasis_l[id] * HammingFilter[id];
}

// pre processing test
// pre-emphasis + framing
__kernel void PreProcessing_t(float PRE_EMPHASIS, int frameLength, 
    int frameStep, int signalLen, int numFrames, 
    __global short* signal, __global float* output,
    __local short* signal_l, __local float* emphasis_l, __local float* cosFactor_l)
{
    // constants
    const int gID_y = get_global_id(0);
    const int gID_x = get_global_id(1);
    const int lID_y = get_local_id(0);
    const int lID_x = get_local_id(1);
    const int groupID = get_group_id(0);

    const emphasisMemLength = get_local_size(1);
    const signalMemLength = emphasisMemLength + 1;

    if (gID_y < numFrames)
    {
        // load signal to local memory. 
        if (lID_x == 0)
        {
            // dump first value
            if (gID_y == 0)
            {
                signal_l[0] = 0;
            }
            else
            {
                signal_l[signalMemLength * lID_y] = signal[frameStep * gID_y - 1];
            }
        }

        if (frameStep * gID_y + gID_x > signalLen)
        {
            signal_l[signalMemLength * lID_y + lID_x + 1] = 0;
        }
        else
        {
            signal_l[signalMemLength * lID_y + lID_x + 1] = signal[frameStep * gID_y + gID_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        //if (gID_y == 260 && get_group_id(1) == 12)
        //{
        //    printf("[%d] - %d\n", gID_x, signal_l[signalMemLength * lID_y + lID_x + 1]);
        //}

        // emphasis

        emphasis_l[emphasisMemLength * lID_y + lID_x] = signal_l[signalMemLength * lID_y + lID_x + 1] -
            PRE_EMPHASIS * signal_l[signalMemLength * lID_y + lID_x];

        // compute cos factor. 
        if (lID_y == 0)
        {
            cosFactor_l[lID_x] = (0.54 - 0.46 * cos(2 * M_PI * gID_x / (frameLength - 1)));
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        output[frameLength * gID_y + gID_x] = emphasis_l[emphasisMemLength * lID_y + lID_x] *
            cosFactor_l[lID_x];

        //if (gID_y == 0 && get_group_id(1) == 0)
        //{
        //    printf("[%d] - %f\n", gID_x, emphasis_l[emphasisMemLength * lID_y + lID_x]);
        //}
    }
}

float2 cMul(const float2 a, const float2 b)
{
    float2 result;

    result.x = (a.x * b.x) - (a.y * b.y);
    result.y = (a.x * b.y) + (a.y * b.x);

    return result;
}

// FFT + Power spectrum
// output size: (N / 2) + 1
__kernel void PowerFFT(int bitLength, int frameLength,
    __global float* input, __global float* output,
    __local float2* input_l, __local float2* output_l,
    __local float2* twiddle)
{
    int id = get_local_id(0);
    const int g = get_group_id(0);

    int N = 1 << bitLength;
    int l = 1 << (bitLength - 1);
    int m = 1;

    int j, k;

    float x;

    __local float2* tmp;

    // copy global data to local memory
    input_l[id].x = input[g * frameLength + id];
    input_l[id].y = 0;

    input_l[id + l].x = input[g * frameLength + (id + l)];
    input_l[id + l].y = 0;

    // calculate twiddle factor
    twiddle[0] = (float2)(1.0f, 0.0f);

    int lo = id + 1;    // location to compute
    int log = 0;        // integer value of log

    while ((lo >>= 1) > 0)
    {
        ++log;
    }
    lo = id + 1;

    int j_t = (lo - (1 << log)) * 2;      // j value of twiddle
    int l_t = 1 << (log + 1);             // l value of twiddle

    x = -M_PI * j_t / l_t;
    twiddle[l_t - 1 + j_t] = (float2)(cos(x), sin(x));

    ++j_t;

    x = -M_PI * j_t / l_t;
    twiddle[l_t - 1 + j_t] = (float2)(cos(x), sin(x));

    barrier(CLK_LOCAL_MEM_FENCE);

    // fft
    for (l = 1 << (bitLength - 1); l >= 1; l >>= 1, m <<= 1)
    {
        j = id / m;
        k = id % m;

        output_l[k + 2 * j * m] = input_l[k + j * m] + input_l[k + j * m + l * m];
        output_l[k + 2 * j * m + m] = cMul(twiddle[l - 1 + j], (input_l[k + j * m] - input_l[k + j * m + l * m]));

        barrier(CLK_LOCAL_MEM_FENCE);

        tmp = input_l;
        input_l = output_l;
        output_l = tmp;
    }

    output[g * ((N / 2) + 1) + id] = (input_l[id].x * input_l[id].x + input_l[id].y * input_l[id].y) / N;

    if (id == 0)
    {
        id = (N / 2);
        output[g * ((N / 2) + 1) + id] = (input_l[id].x * input_l[id].x + input_l[id].y * input_l[id].y) / N;
    }
}

// mel filter bank
__kernel void MelFilBank(int numFrames, int fftLen, int nFilter, int blockSize,
    __global float* powFrames, __global float* fbank,
    __global float* filterBanks,
    __local float* frames_l, __local float* fbank_l)
{
    const int y = get_local_id(0);
    const int x = get_local_id(1);

    const int height = get_local_size(0);
    const int width = get_local_size(1);

    const int gID_y = get_group_id(0);
    const int gID_x = get_group_id(1);

    float value = 0;

    for (int step = 0; step < ceil(((fftLen / (float)blockSize))); ++step)
    {
        // load frames_l
        if ((gID_y * height + y) < numFrames && (step * blockSize + x) < fftLen)
        {
            for (int i = x; i < blockSize; i += width)
            {
                frames_l[y * blockSize + i] = powFrames[(gID_y * height + y) * fftLen + step * blockSize + i];
            }
        }

        // load fbank_l
        if ((step * blockSize + y) < fftLen && (gID_x * width + x) < nFilter)
        {
            for (int i = y; i < blockSize; i += height)
            {
                fbank_l[i * width + x] = fbank[(step * blockSize + i) * nFilter + gID_x * width + x];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < blockSize; ++k)
        {
            value += frames_l[y * blockSize + k] * fbank_l[k * width + x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if ((gID_y * height + y) < numFrames && (gID_x * width + x) < nFilter)
    {
        filterBanks[(gID_y * height + y) + (gID_x * width + x)] = 20 * log10(value);
    }
}

// mel filter bank test
__kernel void MelFilBank_test(int numFrames, int fftLen, int nFilter,
    __global float* powFrames, __global float* fbank, __global float* filterBanks,
    __local float* frames_l)
{
    int y = get_local_id(0);
    int x = get_local_id(1);

    for (int i = y * nFilter + x; i < 16 * fftLen; i += (16 * nFilter))
    {
        frames_l[i] = powFrames[get_group_id(0) * 16 * fftLen + (i / fftLen) * fftLen + i % fftLen];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float value = 0;

    for (int i = 0; i < fftLen; ++i)
    {
        value += frames_l[y * fftLen + i] * fbank[i * nFilter + x];
    }

    filterBanks[get_group_id(0) * 16 * nFilter + y * nFilter + x] = 20 * log10(value);
}

// dct + lifter
__kernel void DCT(int numFrames, int nFilter, int numCeps, float cepLifter,
    __global float* filterBanks, __global float* mfcc,
    __local float* filterBanks_l, __local float* dctFactor)
{
    const int y = get_local_id(0);
    const int x = get_local_id(1);

    const int localSize_y = get_local_size(0);
    const int localSize_x = get_local_size(1);

    const int id = y * localSize_x + x;
    const int gID = get_group_id(0);
    const int localSize = localSize_y * localSize_x;

    // compute dct factor
    //for (int i = id; i < nFilter * numCeps; i += localSize)
    //{
    //    dctFactor[i] = cos(M_PI * (i / nFilter + 1) * (2 * (i % nFilter) + 1) / (2 * nFilter));
    //}

    if (get_global_id(0) < numFrames)
    {
        filterBanks_l[id] = filterBanks[get_global_id(0) * nFilter + x];

        barrier(CLK_LOCAL_MEM_FENCE);

        if (x < numCeps)
        {
            float value = 0;
            for (int k = 0; k < nFilter; ++k)
            {
                value += filterBanks_l[y * localSize_x + k] * dcF[nFilter * x + k];
            }

            float lift = 1 + (cepLifter / 2) * sin(M_PI * x / cepLifter);
            mfcc[numCeps * get_global_id(0) + x] = 2 * value / sqrt(2.0 * nFilter) * lift;
        }
    }
}

// mean normalization
__kernel void MeanNorm(int numFrames, int numCeps,
    __global float* mfcc, __local float* scratch)
{
    const int y_id = get_global_id(0);
    const int x_id = get_global_id(1);

    int local_size = numFrames / 2;

    scratch[y_id] = mfcc[x_id + y_id * numCeps] + mfcc[x_id + (y_id + local_size) * numCeps];

    if (get_local_size(0) * 2 < numFrames)
    {
        if (y_id == 0)
        {
            scratch[y_id] += mfcc[x_id + (numFrames - 1) * numCeps];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int step = local_size >> 1; y_id < step; step >>= 1)
    {
        scratch[y_id] += scratch[y_id + step];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    float mean = scratch[0] / numFrames;

    mfcc[x_id + ((y_id * 2) * numCeps)] /= mean;
    mfcc[x_id + ((y_id * 2 + 1) * numCeps)] /= mean;

    if (y_id == 0)
    {
        mfcc[x_id + (numFrames - 1) * numCeps] /= mean;
    }
}
