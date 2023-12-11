// This program demonstrates vector addition using OpenCL.

// header files

// standard headers
#include<stdio.h>
#include<math.h> // for fabs() - float absolute
#include<stdlib.h> // for exit()

//OpenCL headers
#include<CL/opencl.h>


// global variables
const int iNumberOfArrayElements = 11444777;

// type of platform
cl_platform_id oclPlatformID;

// type of device
cl_device_id oclDeviceId;

// cl_context - state maintaining struct
cl_context oclContext;

cl_command_queue oclCommandQueue;

cl_program oclProgram;

cl_kernel oclKernel;

float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutput = NULL;
float* gold = NULL;

// OpenCL memory object (cl_mem internally void *)
cl_mem deviceInput1 = NULL;
cl_mem deviceInput2 = NULL;
cl_mem deviceOutput = NULL;



// OpenCL Kernel
// __global -> run and call on device 
const char* oclSourceCode =
"__kernel void vectorAdditionGPU(__global float *in1, __global float *in2, __global float *out, int len)" \
"{" \
"int i=get_global_id(0);" \
"if(i < len)" \
"{" \
"out[i]=in1[i]+in2[i];" \
"}" \
"}";

// entry-point function
int main(void) {

    // function declarations
    void fillFloatArrayWithRandomNumbers(float*, int);
    size_t roundGlobalSizeToNearestMultipleOfLocalSize(int, unsigned int);
    void vectorAdditionCPU(const float*, const float*, float*, int);
    void cleanup(void);

    // variable declarations
    int size = iNumberOfArrayElements * sizeof(float);

    cl_int result;

    // code
    // host memory allocation
    hostInput1 = (float*)malloc(size);
    if (hostInput1 == NULL) {
        printf("Host memory allocation failed for hostInput1 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float*)malloc(size);
    if (hostInput2 == NULL) {
        printf("Host memory allocation failed for hostInput2 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutput = (float*)malloc(size);
    if (hostOutput == NULL) {
        printf("Host memory allocation failed for hostOutput array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    gold = (float*)malloc(size);
    if (gold == NULL) {
        printf("Host memory allocation failed for gold array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // filling values into host arrays
    fillFloatArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
    fillFloatArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

    // 1. Get platform's ID (get OpenCL supporting platform's ID)
    /**
     * cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms, cl_uint *num_platforms)
     *
     * num_entries      - The number of entries that can be added to platforms.
     * platforms        - Returns a list of OpenCL platforms found. The cl_platform_id values returned in platforms can be used to
     *                    identify a specific OpenCL platform (like CPU, GPU, Accelerator, etc.)
     * num_platforms    - Returns the number of OpenCL platforms available.
     *
     * Returns CL_SUCCESS if the function is executed successfully, else it returns CL_INVALID_VALUE.
    */
    result = clGetPlatformIDs(1, // 1 - as here we are interested in 1 ID only.
        &oclPlatformID, // give the platformID in this variable.
        NULL); // actual number of platforms found is not important here so this parameter value is NULL.
    if (result != CL_SUCCESS) {
        printf("clGetPlatformIDs() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 2. Get OpenCL supporting GPU device's ID (Obtain the list of devices available on a platform)
    /**
     * cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id *devices, cl_uint *num_devices)
     *
     * platform        - Refers to the platform ID returned by clGetPlatformIDs or can be NULL.
     * device_type     - A bitfield that identifies the type of OpenCL device. It can be used to query specific OpenCL device or all all OpenCL devices available. (like - CL_DEVICE_TYPE_CPU,
     *                  CL_DEVICE_TYPE_GPU, GL_DEVICE_TYPE_ACCELERATOR, etc.)
     * num_entries - The number of device IDs that can be added to devices.
       devices     - A list of OpenCL devices found.
       num_devices - The number of OpenCL devices available that match device_type.
     */
    result = clGetDeviceIDs(oclPlatformID,
        CL_DEVICE_TYPE_GPU,
        1,
        &oclDeviceId,
        NULL);
    if (result != CL_SUCCESS) {
        printf("clGetDeviceIDs() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    //3. Create OpenCL compute context
    oclContext = clCreateContext(NULL, // NULL as the context property param is not needed here.
        1,
        &oclDeviceId,
        NULL, // NULL as no callback function specified here.
        NULL, // NULL para to the callback function since no call function specified here.
        &result);
    if (result != CL_SUCCESS) {
        printf("clCreateContext() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 4. Create command queue
    
    oclCommandQueue = clCreateCommandQueueWithProperties(oclContext, oclDeviceId, NULL, &result);
    if (result != CL_SUCCESS) {
        printf("clCreateCommandQueue() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }


    // 5. Create OpenCL program from .cl
    oclProgram = clCreateProgramWithSource(oclContext, 1, (const char**)&oclSourceCode, NULL, &result);
    if (result != CL_SUCCESS) {
        printf("clCreateProgramWithSource() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 6. Build OpenCL program - Builds (compiles and links) a program executable from the program source or binary.
    result = clBuildProgram(oclProgram, 0, NULL, NULL, NULL, NULL);
    if (result != CL_SUCCESS) {

        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(oclProgram, oclDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("Program build log : %s\n", buffer);
        printf("clBuildProgram() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 7. Create OpenCL kernel by passing kernel function name that we used in .cl file
    oclKernel = clCreateKernel(oclProgram, "vectorAdditionGPU", &result);
    if (result != CL_SUCCESS) {
        printf("clCreateKernel() failed: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 8. Device memory allocation
    deviceInput1 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
    if (result != CL_SUCCESS) {
        printf("clCreateBuffer() failed for 1st input array: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceInput2 = clCreateBuffer(oclContext, CL_MEM_READ_ONLY, size, NULL, &result);
    if (result != CL_SUCCESS) {
        printf("clCreateBuffer() failed for 2nd input array: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    deviceOutput = clCreateBuffer(oclContext, CL_MEM_WRITE_ONLY, size, NULL, &result);
    if (result != CL_SUCCESS) {
        printf("clCreateBuffer() failed for output array: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 9. set 0 based 0th argument i.e. deviceInput1
    result = clSetKernelArg(oclKernel, 0, sizeof(cl_mem), (void*)&deviceInput1);
    if (result != CL_SUCCESS) {
        printf("clSetKernelArg() failed for 1st argument: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0-based 1sth argument i.e. deviceInput2
    result = clSetKernelArg(oclKernel, 1, sizeof(cl_mem), (void*)&deviceInput2);
    if (result != CL_SUCCESS) {
        printf("clSetKernelArg() failed for 2nd argument: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0-based 2nd argument i.e. deviceOutput
    result = clSetKernelArg(oclKernel, 2, sizeof(cl_mem), (void*)&deviceOutput);
    if (result != CL_SUCCESS) {
        printf("clSetKernelArg() failed for 3rd argument: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // set 0-based 3rd argument i.e. len
    result = clSetKernelArg(oclKernel, 3, sizeof(cl_int), (void*)&iNumberOfArrayElements);
    if (result != CL_SUCCESS) {
        printf("clSetKernelArg() failed for 4th argument: %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 10. write above 'input' device buffer to device memory (Enqueue commands to write to a buffer object from host memory.)
    // Simiar to cudaMemcpy()
    result = clEnqueueWriteBuffer(oclCommandQueue, // command-queue
        deviceInput1,  // buffer to write data
        CL_FALSE, // should wait or add commands parallely to the command_queue while writing the data.
        0, // start writing data from 0th byte offset
        size,
        hostInput1,
        0,
        NULL,
        NULL);
    if (result != CL_SUCCESS) {
        printf("clEnqueueWriteBuffer() failed for 1st input device buffer : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = clEnqueueWriteBuffer(oclCommandQueue, deviceInput2, CL_FALSE, 0, size, hostInput2, 0, NULL, NULL);
    if (result != CL_SUCCESS) {
        printf("clEnqueueWriteBuffer() failed for 2nd input device buffer : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // kernel configuration
    size_t local_work_size = 256;
    size_t global_work_size;

    global_work_size = roundGlobalSizeToNearestMultipleOfLocalSize(local_work_size, iNumberOfArrayElements);

    
    result = clEnqueueNDRangeKernel(oclCommandQueue, // command-queue
        oclKernel, // kernel
        1,  //  1-D (dimension of the kernel)
        NULL, // Reserved param
        &global_work_size, // global_work_size
        NULL,
        0,
        NULL,
        NULL);
    if (result != CL_SUCCESS) {
        printf("clEnqueueNDRangeKernel() failed : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // 12. Finish the OpenCL command-queue i.e. allow OpenCL to run all the commands (until this point) in the command queue.
    clFinish(oclCommandQueue);

    

    // 13. Read back result from the device (i.e. from deviceOutput) into CPU variable (i.e. hostOutput)
    result = clEnqueueReadBuffer(oclCommandQueue, deviceOutput, CL_TRUE, 0, size, hostOutput, 0, NULL, NULL);
    if (result != CL_SUCCESS) {
        printf("clEnqueueReadBuffer() failed : %d\n", result);
        cleanup();
        exit(EXIT_FAILURE);
    }

    // vector addition on host
    vectorAdditionCPU(hostInput1, hostInput2, gold, iNumberOfArrayElements);

    // comparison
    const float epsilon = 0.000001f;
    int breakValue = -1;
    bool bAccuracy = true;
    for (int i = 0; i < iNumberOfArrayElements; i++) {
        float val1 = gold[i];
        float val2 = hostOutput[i];
        if (fabs(val1 - val2) > epsilon) {
            bAccuracy = false;
            breakValue = i;
            break;
        }
    }

    char str[128];
    if (bAccuracy == false) {
        sprintf_s(str, "Comparison of CPU and GPU vector addition is not within accuracy of 0.000001f at array index %d", breakValue);
    }
    else {
        sprintf_s(str, "Comparison of CPU and GPU vector addition is within accuracy of 0.000001f");
    }

    // output
    printf("Array1 begins from 0th index %.6f to %dth index %.6f\n", hostInput1[0], iNumberOfArrayElements - 1, hostInput1[iNumberOfArrayElements - 1]);
    printf("Array2 begins from 0th index %.6f to %dth index %.6f\n", hostInput2[0], iNumberOfArrayElements - 1, hostInput2[iNumberOfArrayElements - 1]);
    printf("OpenCL Kernel Global Work Size = %lu and Local Work Size = %lu\n", global_work_size, local_work_size);
    printf("Output array begins from 0th index %.6f to %dth index %.6f\n", hostOutput[0], iNumberOfArrayElements - 1, hostOutput[iNumberOfArrayElements - 1]);
    printf("%s\n", str);

    // cleanup
    cleanup();

    return(0);
}

void fillFloatArrayWithRandomNumbers(float* arr, int len) {

    // code
    const float fscale = 1.0f / (float)RAND_MAX;
    for (int i = 0; i < len; i++) {
        arr[i] = fscale * rand();
    }
}

void vectorAdditionCPU(const float* arr1, const float* arr2, float* out, int len) {

    // code
    

    for (int i = 0; i < len; i++) {
        out[i] = arr1[i] + arr2[i];
    }

    
}

size_t roundGlobalSizeToNearestMultipleOfLocalSize(int local_size, unsigned int global_size) {

    // code
    unsigned int r = global_size % local_size;
    if (r == 0) {
        return (global_size);
    }
    else {
        return (global_size + local_size - r);
    }
}

void cleanup() {

    // code
    if (deviceOutput) {
        clReleaseMemObject(deviceOutput);
        deviceOutput = NULL;
    }

    if (deviceInput2) {
        clReleaseMemObject(deviceInput2);
        deviceInput2 = NULL;
    }

    if (deviceInput1) {
        clReleaseMemObject(deviceInput1);
        deviceInput1 = NULL;
    }

    if (oclKernel) {
        clReleaseKernel(oclKernel);
        oclKernel = NULL;
    }

    if (gold) {
        free(gold);
        gold = NULL;
    }

    if (oclProgram) {
        clReleaseProgram(oclProgram);
        oclProgram = NULL;
    }

    if (oclCommandQueue) {
        clReleaseCommandQueue(oclCommandQueue);
        oclCommandQueue = NULL;
    }

    if (oclContext) {
        clReleaseContext(oclContext);
        oclContext = NULL;
    }

    if (hostOutput) {
        free(hostOutput);
        hostOutput = NULL;
    }

    if (hostInput2) {
        free(hostInput2);
        hostInput2 = NULL;
    }

    if (hostInput1) {
        free(hostInput1);
        hostInput1 = NULL;
    }
}