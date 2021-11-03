# Collaborative Document. Day 2, November 3
2021-11-03-ds-gpu

Welcome to The Workshop Collaborative Document 
 

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents. 

All content is publicly available under the Creative Commons Attribution License 

https://creativecommons.org/licenses/by/4.0/ 

 ---------------------------------------------------------------------------- 

This is the Document for today: [link](https://tinyurl.com/48mfnt44)

Collaborative Document day 1: [link](https://tinyurl.com/2v5ycpnz)

Collaborative Document day 2: [link](https://tinyurl.com/48mfnt44) 
  

## 👮Code of Conduct 

* Participants are expected to follow those guidelines: 
* Use welcoming and inclusive language 
* Be respectful of different viewpoints and experiences 
* Gracefully accept constructive criticism 
* Focus on what is best for the community 
* Show courtesy and respect towards other community members 
 

## ⚖️ License 

All content is publicly available under the Creative Commons Attribution License: https://creativecommons.org/licenses/by/4.0/ 

 

## 🙋Getting help 

to ask a question, type `/hand` in the chat window 

to get help, type `/help` in the chat window 

you can ask questions in the document or chat window and helpers will try to help you 
 

## 🖥 Workshop website 

* [Course](https://escience-academy.github.io/2021-11-02-ds-gpu/)
* [JupyterHub](https://jupyter.lisa.surfsara.nl/jhlsrf009/)
* [Google Colab](https://colab.research.google.com)
* [Post-workshop survey](https://www.surveymonkey.com/r/WQLWBYW)
 

## 👩‍🏫👩‍💻🎓 Instructors 

Alessio Sclocco, Ben van Werkhoven, Hanno Spreeuw
 

## 🧑‍🙋 Helpers 

Victor Azizi, Johan Hidding
 

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call 

Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city 

### Instant feedback

## 🗓️ Agenda 

09:00 	Welcome and icebreaker

09:15 	Introduction to CUDA

10:15 	Coffee break

10:30 	CUDA memories and their use

11:30 	Coffee break

11:45 	Data sharing and synchronization

12:45 	Wrap-up and post-workshop survey

13:00 	END 

## 🧠 Collaborative Notes 
Next step from yesterday. We are still working from Python, however we'll be writing the GPU part of the computation in CUDA. If you have a GPU that is not from Nvidia you don't have CUDA, but the concept we discuss apply to OpenCL as well. We'll start with an example where we add two arrays of numbers:

```python=
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
    return C
```

In CUDA this looks like

```c
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
```

CUDA functions in general:

- Do not return a value
- Have an extra keyword:
    -  `__global__` meaning: callable from host
    -  `__device__` meaning: only callable from GPU
    -  `__host__` meaning: explicit host function
- a `threadIdx` variable substituting a for-loop.

Let's check that our codes do the same thing:

```python=
size = 1024
a_gpu = cupy.random.rand(size, dtype=cupy.float32)  # single precision
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
```

We need to give the previous CUDA code to the CUDA compiler.

```python=
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
'''
```

```python=
vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")
vector_add_gpu((1,1,1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```

Yesterday we talked about a block of threads (2nd argument). We are providing CUDA with the size of the problem here. The first argument is the number of blocks (in three dimensions), in this case 1 block, 1024 threads.

We want to check that our CUDA function gives the correct result:

```python=
import numpy as np

a_cpu = cupy.asnumpy(a_gpu)
b_cpu = cupy.asnumpy(b_gpu)
c_cpu = np.zeros_like(a_cpu)

vector_add(a_cpu, b_cpu, c_cpu, size)
np.allclose(c_cpu, c_gpu)
```

### Using thread blocks
Let's change the size of the vector:

```python=
size = 2048

a_gpu = cupy.random.rand(size, dtype=cupy.float32)  # single precision
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)
vector_add_gpu((1,1,1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```

Ooops!

```
---------------------------------------------------------------------------
CUDADriverError                           Traceback (most recent call last)
/tmp/ipykernel_15883/880747844.py in <module>
      4 b_gpu = cupy.random.rand(size, dtype=cupy.float32)
      5 c_gpu = cupy.zeros(size, dtype=cupy.float32)
----> 6 vector_add_gpu((1,1,1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))

<... snip ...>

CUDADriverError: CUDA_ERROR_INVALID_VALUE: invalid argument
```

We exceeded the maximum number of threads per block.

> Break time at 10:15, back at 10:30

We can change the number of thread blocks to get to sizes larger than 1024.

```python=
vector_add_gpu((2,1,1), (size // 2, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```

CUDA is happy now, but our test fails. We need to change the CUDA code to take both the `threadIdx` and the block index `blockIdx` into account (see challenge: scaling up):

```python=
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    C[item] = A[item] + B[item];
}
'''
```

We can also ask for the dimensions (all of these are 3d variables, having `x`, `y` and `z` members):
- `blockDim` dimension of the blocks, ie number of threads per block
- `gridDim` number of blocks in grid
- `blockIdx` index of the block into the grid
- `threadIdx` index of thread within current block

When we call the kernel from Python these values are connected as follows:

```python=
my_kernel((gridDim.x, gridDim.y, gridDim.z), (blockDim.x, blockDim.y, blockDim.z), ...)
```

The values for `blockIdx.x` and `threadIdx.x` will vary for every thread running in the ranges `[0:gridDim.x]` and `[0:blockDim.x]` (and similar for `y` and `z` directions).

### Bounds checking
We have the `size` parameter in our kernel. We can use this to do bounds checking. This way we can write to arrays that have a size that is not exactly a multiple of our `blockDim`.

```c=
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item < size)
    {
        C[item] = A[item] + B[item];        
    }
}
```

To call our kernel in a more generic way, we parametrize the `grid_size` and `block_size`.

```python=
import math
vector_add_gpu = cupy.RawKernel(vector_add_cuda_code, "vector_add")
threads_per_block = 1024
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)
vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))
```

By using the `math.ceil` function we guarantee that we compute enough blocks to cover the entire array. Now if we change the `size` variable, the code should be resilient and compute the right result.

### Memory
When we look at the `vector_add` kernel, we see that we have several input variables and some variables that we use inside the kernel. We may introduce more variables inside the kernel, but those are not visible outside the CUDA code.

```c=
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    float factor = 2.0;
    if (item < size)
    {
        C[item] = (A[item] + B[item]) * factor;
    }
    
    factor = factor + 1;
}
```

In this example `factor` stays `2.0` always! (No different from normal C or Python). If we look at the value of `*C` however, this memory is *persistent*. Changes we make affect execution *globally*. The pointer `*C` lives in **global memory**.


Suppose now that we write to an array:

```c=
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    float *D[1024];
    if (item < size)
    {
        C[item] = (A[item] + B[item]) * factor;
        D[threadIdx.x] = A[item];
    }
}
```

In this case `*D` still lives privately inside our kernel. This can be useful if we need some private memory to do computations. This memory is still not persistent over different threads.

If we want to share data between threads, say for communication, we need to change the code a little.

> We do the primes exercise, at 11:30 break until 11:45

#### Shared memory
We can have shared memory.

```c=
extern "C"
__global__ void vector_add(const float *A, const float *B, float *C, const int size)
{
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float temp[3];

    if (item < size)
    {
        temp[0] = A[item];
        temp[1] = B[item];
        C[item] = temp[0] + temp[1];
    }
}
```

This will give incorrect results because all threads will be writing to `temp` at the same time. The concept of shared memory is extremely useful though! Note that the size of the shared array needs to be known at compile time. Suppose we need twice the block size in shared memory, we cannot say:

```c=
__shared__ float temp[blockDim.x * 2];
```

Even though we know that `blockDim` is usually constant. To solve this, we can tell CUDA that we will take care of allocating memory our selves. Steps: remove the dimension, add the `extern` keyword.

```c=
extern __shared__ float temp[];
```

Now we need to add a parameter to the kernel call in python:

```python=
float_size = cupy.dtype(cpuy.float32).itemsize   # fancy way of saying 4
shared_size = thread_per_block * 2 * float_size
vector_add_gpu(
    grid_size, block_size, (a_gpu, b_gpu, c_gpu, size),
    shared_mem=shared_size)
```

You can have multiple shared arrays, but it takes doing some pointer arithmetic inside your kernel.

#### Example: histogram function

```python=
import numpy as np

def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] += 1
    return output_array

input_array = np.random.randint(256, size=2048, dtype=np.int32)
output_array = np.zeros(256, dtype=np.int32)
output_array = histogram(input_array, output_array)
```

```c=
extern "C"
__global__ void histogram(const int * input, int *output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    output[input[item]] = output[input[item]] + 1;
}
```

This will not work. We have race conditions, reading and writing to `output` simultaneously. We may solve this using **atomics**.

```c=
__global__ void histogram(const int * input, int *output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    atomicAdd(&(output[input[item]]), 1);
}
```

To run this:

```python=
import math
import numpy
import cupy

size = 2048
input_gpu = cupy.random.randint(256, size=size, dtype=cupy.int32)
input_cpu = cupy.asnumpy(input_gpu)
output_gpu = cupy.zeros(256, dtype=cupy.int32)
output_cpu = cupy.asnumpy(output_gpu)

histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int *output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    atomicAdd(&(output[input[item]]), 1);
}
'''

histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

histogram(input_cpu, output_cpu)
histogram_gpu(grid_size, block_size, (input_gpu, output_gpu))

numpy.allclose(output_cpu, output_gpu)
```

Atomics are very expensive. This way the GPU is not doing very much in parallel.

We can improve the memory access pattern as follows:

```c=
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    extern __shared__ int temp_histogram[];
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
```

## Exercises

### Challenge: Loose threads

We know enough now to pause for a moment and do a little exercise. Assume that in our `vector_add` kernel we replace the following line:

```
int item = threadIdx.x;
```

With this other line of code:

```
int item = 1;
```

What will the result of this change be?

1) Nothing changes
2) Only the first thread is working
3) Only `C[1]` is written
4) All elements of `C` are zero

### Challenge: Scaling up

In the following code, fill in the blank to work with vectors that are larger than the largest CUDA block (i.e. 1024).

```
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = ______________;
   C[item] = A[item] + B[item];
}
```

### Challenge: Compute prime numbers with CUDA

Given the following Python code, similar to what we have seen in the previous episode about Numba, write the missing CUDA kernel that computes all the prime numbers up to a certain upper bound.

```
# CPU
def all_primes_to(upper : int, prime_list : list):
    for num in range(2, upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            prime_list[num] = 1

upper_bound = 100000
all_primes_cpu = numpy.zeros(upper_bound, dtype=numpy.int32)
all_primes_cpu[0] = 1
all_primes_cpu[1] = 1
%timeit all_primes_to(upper_bound, all_primes_cpu)

# GPU
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
}
'''
# Allocate memory
all_primes_gpu = cupy.zeros(upper_bound, dtype=cupy.int32)

# Compile and execute code
all_primes_to_gpu = cupy.RawKernel(check_prime_gpu_code, "all_primes_to")
grid_size = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size = (1024, 1, 1)
%timeit all_primes_to_gpu(grid_size, block_size, (upper_bound, all_primes_gpu))

# Test
if numpy.allclose(all_primes_cpu, all_primes_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
```

Solution:

```c=
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
    int number = blockIdx.x * blockDim.x + threadIdx.x;
    int result = 1;
    
    if (number < size)
    {
        for (int factor = 2; factor <= number / 2; ++factor)
        {
            if (number % factor == 0)
            {
                result = 0;
                break;
            }
        }
        all_prime_numbers[number] = result;
    }
}
```

### Challenge: use shared memory to speed up the histogram

Implement a new version of the `histogram` function that uses shared memory.

Hint: try to reduce conflicts, and improve the memory access pattern.
Hint: for this exercise, assume that the size of output is the same as the number of threads in a block.


## Open Questions 

- Hi, I have an error with the first function
    ```python
    __global__ void vector_add(const float *A, const float *B, float *C, const int size)
    {
        int item = threadIdx.x;
        C[item] = A[item] + B[item];
    }
    ```
    The error says the following:
    ```File "<ipython-input-2-88cf2f175a29>", line 2
    __global__ void vector_add(const float *A, const float *B, float *C, const int size)
               ^
    SyntaxError: invalid syntax
    ```
    - The thing is that you are now passing CUDA code to the Python interpreter. CUDA is a different programming language, and we are using CuPy to compile CUDA code on the GPU. To do so, we pass the CUDA code as a string to a Cupy function that calls the NVRTC compiler. So instead of passing the CUDA code to the Python interpreter we need to pass it as a string to the RawKernel function of cupy. Please let me know if this solves your problem.
    - That works. Thanks Ben

- I have a error too:

    ![](https://i.imgur.com/FEtl0aF.png)

    - it appears 'size' that you use now is larger than what was used to create the arrays A, B, and C. If these are numpy arrays you could just use C=A+B to do a pointwise addition of all elements in A and B and store the result in C. 
    - double check with a_cpu.shape, b_cpu.shape and c_cpu.shape
      The size = 1024. Shape = (1024,)
    - The error is actually in the line `C = []` that shouldn't be there. You're assigning an empty list to the variable C.
    - indeed, good catch Johan!
    - good catch indeed! Thanks!

## 📚 Resources 

* [Upcoming eScience Center workshops](https://www.esciencecenter.nl/digital-skills/)
* [HackMD Markdown Guide](https://www.markdownguide.org/tools/hackmd/)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [NL-RSE meetup December 1st](https://www.eventbrite.co.uk/e/nl-rse-meetup-december-1-2021-tickets-195099246097)
* [Post-workshop survye](https://www.surveymonkey.com/r/WQLWBYW)
* [eScience Center GPU Course](https://github.com/benvanwerkhoven/gpu-course)
