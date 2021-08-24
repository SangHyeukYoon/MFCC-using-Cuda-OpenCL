#include "clUtils.h"

cl_platform_id CreatePlatform()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id* platformIds;
    cl_context context = NULL;

    // ��ü �÷��� ���� �����Ѵ�. 
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;

        return NULL;
    }

    // ��ġ�� �÷����� ���� �޸𸮸� �Ҵ��Ѵ�. 
    platformIds = (cl_platform_id*)alloca(sizeof(cl_platform_id) * numPlatforms);

    // �÷��� ID������ ���Ǹ� �����Ѵ�. 
    errNum = clGetPlatformIDs(numPlatforms, platformIds, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;

        return NULL;
    }

    std::cout << "Platform Spec" << std::endl;
    std::cout << "Number of platforms: \t" << numPlatforms << std::endl << std::endl;

    // �÷��� ����� �ݺ��ϸ鼭 ������ ������ �����ش�. 
    for (cl_uint i = 0; i < numPlatforms; ++i)
    {
        std::cout << std::string(64, '=') << std::endl;
        std::cout << "Platform:\t\t" << i << std::endl;

        DisplayPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE");
        DisplayPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION");
        DisplayPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
        //DisplayPlatformInfo(platformIds[i], CL_PLATFORM_EXTENSIONS, "CL_PLATFORM_EXTENSIONS");

        std::cout << std::string(64, '=') << std::endl << std::endl;
    }

    // ù��° �÷����� ��ȯ�Ѵ�. 
    return platformIds[0];
}

void DisplayPlatformInfo(cl_platform_id id, cl_platform_info name, std::string str)
{
    cl_int errNum;
    std::size_t paramValueSize;

    errNum = clGetPlatformInfo(id, name, 0, NULL, &paramValueSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL platform " << str << "." << std::endl;

        return;
    }

    char* info = (char*)alloca(sizeof(char) * paramValueSize);

    errNum = clGetPlatformInfo(id, name, paramValueSize, info, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL platform " << str << "." << std::endl;

        return;
    }

    std::cout << str << ":\t" << info << std::endl;
}

cl_device_id CreateDevice(cl_platform_id platform)
{
    cl_int errNum;
    cl_uint numDevices;
    cl_device_id* deviceIds;

    // ��ü ����̽� ���� �����Ѵ�. 
    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (errNum != CL_SUCCESS || numDevices < 1)
    {
        std::cerr << "No GPU device found for platform " << platform << std::endl;

        return NULL;
    }

    // ��ġ�� ����̽��� ���� �޸𸮸� �Ҵ��Ѵ�. 
    deviceIds = (cl_device_id*)alloca(sizeof(cl_device_id) * numDevices);

    // ����̽� ID������ �����Ѵ�. 
    errNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, deviceIds, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "No GPU device found for platform " << platform << std::endl;

        return NULL;
    }

    std::cout << "Device Spec" << std::endl;
    std::cout << "Number of Devices: \t" << numDevices << std::endl << std::endl;

    // ����̽��� ����� �ݺ��ϸ� ������ ������ �����ش�. 
    for (cl_uint i = 0; i < numDevices; ++i)
    {
        std::cout << std::string(64, '=') << std::endl;
        std::cout << "Device:\t\t\t\t\t" << i << std::endl;

        DisplayDeviceInfo(deviceIds[i]);

        std::cout << std::string(64, '=') << std::endl << std::endl;
    }

    return deviceIds[0];
}

void DisplayDeviceInfo(cl_device_id id)
{
    cl_int errNum;

    // device name
    std::size_t paramValueSize;

    errNum = clGetDeviceInfo(id, CL_DEVICE_NAME, 0, NULL, &paramValueSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_NAME" << "." << std::endl;

        return;
    }

    char* deviceName = (char*)alloca(sizeof(char) * paramValueSize);

    errNum = clGetDeviceInfo(id, CL_DEVICE_NAME, paramValueSize, deviceName, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_NAME" << "." << std::endl;

        return;
    }

    std::cout << "CL_DEVICE_NAME" << ":\t\t\t\t" << deviceName << std::endl;

    // device vendor
    errNum = clGetDeviceInfo(id, CL_DEVICE_VENDOR, 0, NULL, &paramValueSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_VENDOR" << "." << std::endl;

        return;
    }

    char* deviceVendor = (char*)alloca(sizeof(char) * paramValueSize);

    errNum = clGetDeviceInfo(id, CL_DEVICE_VENDOR, paramValueSize, deviceVendor, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_VENDOR" << "." << std::endl;

        return;
    }

    std::cout << "CL_DEVICE_VENDOR" << ":\t\t\t" << deviceVendor << std::endl;

    // device version
    errNum = clGetDeviceInfo(id, CL_DEVICE_VERSION, 0, NULL, &paramValueSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_VERSION" << "." << std::endl;

        return;
    }

    char* deviceVersion = (char*)alloca(sizeof(char) * paramValueSize);

    errNum = clGetDeviceInfo(id, CL_DEVICE_VERSION, paramValueSize, deviceVersion, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_VERSION" << "." << std::endl;

        return;
    }

    std::cout << "CL_DEVICE_VERSION" << ":\t\t\t" << deviceVersion << std::endl;

    // max compute units
    cl_uint maxCmpUnits;

    errNum = clGetDeviceInfo(id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxCmpUnits, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_MAX_COMPUTE_UNITS" << "." << std::endl;

        return;
    }

    std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS:\t\t" << maxCmpUnits << std::endl;

    // max work item dimensions
    // ������ ���� ���� �𵨿��� ����ϴ� 
    // work-item ID�� ���ϴ� ������ �ִ�
    cl_uint maxWID;
    errNum = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWID, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS" << "." << std::endl;

        return;
    }

    std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t" << maxWID << std::endl;

    // max work item size
    // clEnqueueNDRangeKernel���� work-group�� �� ������ ���� 
    // work-item�� �ִ� ũ��
    std::size_t* maxWIS = (std::size_t*)alloca(sizeof(std::size_t) * maxWID);
    errNum = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(std::size_t) * maxWID, maxWIS, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_MAX_WORK_ITEM_SIZES" << "." << std::endl;

        return;
    }

    std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t(";
    for (cl_uint i = 0; i < maxWID; ++i)
    {
        std::cout << maxWIS[i];

        if (i + 1 < maxWID)
        {
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;

    // max work group size
    // ������ ���� ���� �𵨿��� ����ϴ� �� Ŀ���� ������ ��
    // �� work-group�� �ִ� work-item���� �ִ� ����
    std::size_t maxWGS;
    errNum = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(std::size_t), &maxWGS, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_MAX_WORK_GROUP_SIZE" << "." << std::endl;

        return;
    }

    std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE:\t\t" << maxWGS << std::endl;

    // global memory size
    cl_ulong glbMemSize;
    errNum = clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &glbMemSize, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_GLOBAL_MEM_SIZE" << "." << std::endl;

        return;
    }

    std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE:\t\t" << glbMemSize << " bytes" << std::endl;

    // global memory cache size
    cl_ulong glbMemCacheSize;
    errNum = clGetDeviceInfo(id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &glbMemCacheSize, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE" << "." << std::endl;

        return;
    }

    std::cout << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:\t" << glbMemCacheSize << " bytes" << std::endl;

    // local memory size
    cl_ulong locMemSize;
    errNum = clGetDeviceInfo(id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &locMemSize, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_LOCAL_MEM_SIZE" << "." << std::endl;

        return;
    }

    std::cout << "CL_DEVICE_LOCAL_MEM_SIZE:\t\t" << locMemSize << " bytes" << std::endl;

    // max constant buffer size
    cl_ulong maxCBS;
    errNum = clGetDeviceInfo(id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &maxCBS, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed to find OpenCL device " << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE" << "." << std::endl;

        return;
    }

    std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t" << maxCBS << " bytes" << std::endl;
}

cl_context CreateContext(cl_platform_id platform, cl_device_id device)
{
    cl_int errNum;
    cl_context context = NULL;

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = clCreateContext(contextProperties, 1, &device, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create context." << std::endl;

        return NULL;
    }

    return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device)
{
    cl_command_queue commandQueue = NULL;

    commandQueue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);
    if (commandQueue == NULL)
    {
        std::cerr << "Failed to create commandQueue for device 0." << std::endl;

        return NULL;
    }

    return commandQueue;
}

cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;

        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char* srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1, (const char**)&srcStr, NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;

        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // ������ ���� ������ �����Ѵ�.
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);

        return NULL;
    }

    return program;
}

cl_program CreateProgramFromBinary(cl_context context, cl_device_id device, const char* fileName)
{
    FILE* fp = fopen(fileName, "rb");
    if (fp == NULL)
    {
        return NULL;
    }

    // ���̳ʸ��� ũ�⸦ �����Ѵ�. 
    size_t binarySize;
    fseek(fp, 0, SEEK_END);
    binarySize = ftell(fp);
    rewind(fp);

    // ��ũ���� ���̳ʸ��� �����Ѵ�. 
    unsigned char* programBinary = new unsigned char[binarySize];
    fread(programBinary, 1, binarySize, fp);
    fclose(fp);

    cl_int errNum = 0;
    cl_program program;
    cl_int binaryStatus;

    program = clCreateProgramWithBinary(context, 1, &device, &binarySize,
        (const unsigned char**)&programBinary, &binaryStatus, &errNum);

    delete[] programBinary;

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error loading program binary." << std::endl;

        return NULL;
    }

    if (binaryStatus != CL_SUCCESS)
    {
        std::cerr << "Invalid binary for device." << std::endl;

        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // ���� ������ ����Ѵ�. 
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in program: " << std::endl;
        std::cerr << buildLog << std::endl;

        clReleaseProgram(program);

        return NULL;
    }

    return program;
}

bool SaveProgramBinary(cl_program program, cl_device_id device, const char* fileName)
{
    cl_uint numDevices = 0;
    cl_int errNum;

    // ���α׷��� ������ ����̽����� ���� �����Ѵ�. 
    errNum = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for number of devices." << std::endl;

        return false;
    }

    // ��ü ����̽� ID���� ���Ѵ�. 
    cl_device_id* devices = new cl_device_id[numDevices];
    errNum = clGetProgramInfo(program, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * numDevices,
        devices, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for devices." << std::endl;

        delete[] devices;

        return false;
    }

    // �� ���α׷� ���̳ʸ��� ũ�⸦ �����Ѵ�. 
    size_t* programBinarySizes = new size_t[numDevices];
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t) * numDevices,
        programBinarySizes, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for program binary sizes." << std::endl;

        delete[] devices;
        delete[] programBinarySizes;

        return false;
    }

    unsigned char** programBinaries = new unsigned char* [numDevices];
    for (cl_uint i = 0; i < numDevices; ++i)
    {
        programBinaries[i] = new unsigned char[programBinarySizes[i]];
    }

    // ��� ���α׷��� ���̳ʸ��� ���Ѵ�. 
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * numDevices,
        programBinaries, NULL);

    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error querying for program binary sizes." << std::endl;

        delete[] devices;
        delete[] programBinarySizes;
        for (cl_uint i = 0; i < numDevices; ++i)
        {
            delete[] programBinaries[i];
        }
        delete[] programBinaries;

        return false;
    }

    // ��û�� ����̽��� ���� ���̳ʸ��� ��ũ�� �����Ѵ�. 
    for (cl_uint i = 0; i < numDevices; ++i)
    {
        if (devices[i] == device)
        {
            FILE* fp = fopen(fileName, "wb");
            fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
            fclose(fp);

            break;
        }
    }

    // �����ϱ�
    delete[] devices;
    delete[] programBinarySizes;
    for (cl_uint i = 0; i < numDevices; ++i)
    {
        delete[] programBinaries[i];
    }
    delete[] programBinaries;

    return true;
}
