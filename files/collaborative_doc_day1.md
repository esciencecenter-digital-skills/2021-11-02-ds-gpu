# Collaborative Document. Day 1, November 2 

2021-11-02-ds-gpu

Welcome to The Workshop Collaborative Document 
 

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents. 

All content is publicly available under the Creative Commons Attribution License 

https://creativecommons.org/licenses/by/4.0/ 

 ---------------------------------------------------------------------------- 

This is the Document for today: [link](https://tinyurl.com/2v5ycpnz)

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
 

## 👩‍🏫👩‍💻🎓 Instructors 

Alessio Sclocco, Ben van Werkhoven, Hanno Spreeuw
 

## 🧑‍🙋 Helpers 

Victor Azizi, Johan Hidding
 

## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call 

Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city 

## 🗓️ Agenda 

09:00 	Welcome and icebreaker

09:15 	Introduction

09:30 	Convolve an image with a kernel on a GPU using CuPy

10:15 	Coffee break

10:30 	Running CPU/GPU agnostic code using CuPy

11:30 	Coffee break

11:45 	Run your Python code on a GPU using Numba

12:45 	Wrap-up

13:00 	END
 

## 🧠 Collaborative Notes 

### Introduction

Alessio's example

```python
import numpy as np
size = 4096 * 4096
input = np.random.random(size).astype(np.float32)
```

```python
%timeit output = np.sort(input)
```

```python
import cupy as cp
input_gpu = cp.asarray(input)
```

```python
%timeit output_gpu = cp.sort(input_gpu)
```

### Convolution

[Wikipedia page on Convolution](https://en.wikipedia.org/wiki/Convolution)

```python
import cupy as cp
import numba as nb
import numpy as np
```

```python
tile = np.zeros((16, 16))
tile[8, 8] = 1
import pylab as pyl
%matplotlib inline
pyl.imshow(tile)
pyl.show()
```

```python
deltas = np.tile(tile, (128, 128))
print(deltas.shape)
```

```python
pyl.imshow(deltas[0:48, 0:48])
pyl.show()
```

```python
axis_point = np.linspace(-2, 2, 15)
x, y = np.meshgrid(axis_point, axis_point)
gauss = np.exp(-x**2 - y**2)
pyl.imshow(gauss)
pyl.colorbar()
pyl.show()
```

```python
from scipy.signal import convolve2d as convolve2d_cpu

convolved_image_using_cpu = convolve2d_cpu(deltas, gauss)
pyl.imshow(convolved_image_using_cpu[0:48, 0:48])
pyl.show()
```

```python
%timeit convolve2d_cpu(deltas, gauss)
```

![CPU / GPU System](https://carpentries-incubator.github.io/lesson-gpu-programming/fig/CPU_and_GPU_separated.png "CPU / GPU System")

```python
deltas_gpu = cp.asarray(deltas)
gauss_gpu = cp.asarray(gauss)
```

```python
from cupyx.scipy.signal import convolve2d as convolve2d_gpu

convolved_image_using_gpu = convolve2d_gpu(deltas_gpu, gauss_gpu)
%timeit convolve2d_gpu(deltas_gpu, gauss_gpu)
```

```python
image_convolved_on_gpu_copied_back_to_cpu = cp.asnumpy(convolved_image_using_gpu)
pyl.imshow(image_convolved_on_gpu_copied_back_to_cpu[0:48, 0:48])
```

```python
pyl.imshow(convolved_image_using_gpu.get()[0:48, 0:48])
```

```python
np.allclose(image_convolved_on_gpu_copied_back_to_cpu, convolved_image_using_cpu)
np.allclose(convolved_image_using_gpu, convolved_image_using_cpu)
```

```python
deltas_1d_gpu = cp.ravel(deltas_gpu)
gauss_1d_gpu = cp.ravel(gauss_gpu)
%timeit np.convolve(deltas_1d_gpu, gauss_1d_gpu)
```

```python
deltas_1d = np.ravel(deltas)
gauss_1d = np.ravel(gauss)
%timeit (deltas_1d, gauss_1d)
```

### Compute prime numbers

```python
def find_all_primes(max):
    all_primes = []
    for i in range(2, max):
        for j in range(2, (i // 2) + 1):
            if (i % j) == 0:
                break
        else:
            all_primes.append(i)
    return all_primes
```

```python
%timeit find_all_primes(10000)
```

```python
import numba as nb
find_all_primes_jit_compiled = nb.jit(nopython=True)(find_all_primes)
%timeit find_all_primes_jit_compiled(10000)
```

```python
@nb.jit(nopython=True)
def find_all_primes(max):
    all_primes = []
    for i in range(2, max):
        for j in range(2, (i // 2) + 1):
            if (i % j) == 0:
                break
        else:
            all_primes.append(i)
    return all_primes
```

![](https://i.imgur.com/FlIZcFH.png)

```python
from numba import int32

@nb.vectorize([int32(int32)], target='cpu')
def check_if_this_is_a_prime(i):
    for j in range(2, (i // 2) + 1):
        if (i % j) == 0:
            return 0
    else:
        return i
```

```python
check_if_this_is_a_prime(np.arange(2, 100, dtype=np.int32))
```

```python
%timeit check_if_this_is_a_prime(np.arange(2, 10000, dtype=np.int32))
```

```python
from numba import cuda

@cuda.jit
def find_prime_on_gpu(i, result):
    result[0] = 0
    for j in range(2, (i // 2) + 1):
        if (i % j) == 0:
            break
        else:
            result[0] = i
```

```python
result = np.ones(1, dtype=np.int32)
find_prime_on_gpu[1, 1](101, result)
print(result)
```

```python
def find_all_primes_using_both_cpu_and_gpu(max):
    all_prime_numbers = []
    for i in range(2, max):
        result = np.ones(1, dtype=np.int32)
        find_prime_on_gpu[1, 1](i, result)
        if result[0] != 0:
            all_prime_numbers.append(result[0])
    return all_prime_numbers

find_all_primes_using_both_cpu_and_gpu(100)
```

```python
%timeit find_all_primes_using_both_cpu_and_gpu(10000)
```

```python
@nb.vectorize([int32(int32)], target='cuda')
def check_if_this_is_a_prime(i):
    for j in range(2, (i // 2) + 1):
        if (i % j) == 0:
            return 0
    else:
        return i

%timeit check_if_this_is_a_prime(np.arange(2, 10000, dtype=np.int32))
```

## Exercises

### Challenge: fairer runtime comparison CPU vs. GPU

Compute the CPU vs GPU speedup while taking into account the transfers of data to the GPU and back. You should now find a lower speedup from taking the overhead of the transfer of arrays into account. Hint: To copy a CuPy array back to the host (CPU), use `cp.asnumpy()`.

```python
def convolve_using_gpu(img, kernel):
    img_gpu = cp.asarray(img)
    kernel_gpu = cp.asarray(kernel)
    result = convolve2d_gpu(img_gpu, kernel_gpu)
    img = cp.asnumpy(result)
    return img

def convolve_using_cpu(img, kernel):
    result = convolve2d_cpu(img, kernel)
    return result


time_cpu = %timeit -o convolve_using_cpu(deltas, gauss_kernel)
time_gpu = %timeit -o convolve_using_gpu(deltas, gauss_kernel)
speedup = time_cpu.best / time_gpu.best
print(speedup)
```
### Challenge: compute prime numbers

Write a function `find_all_primes_cpu_and_gpu` that uses `find_prime_on_gpu` and the outer loop similar to `find_all_primes`.
How long does it take to find all primes up to 10000?

## Open Questions
- I ran into the following error:
![](https://i.imgur.com/GUh0LpB.png)
    - ANSWER: watch the capitalization convolve2D_cpu vs convolve2d_cpu
    - YES: The D should be a d in convolve2D_cpu
- Maybe we will discuss this later, but what if I have an image that don't fit in GPU memory? Is there a way of automatically stream the memory from Host->GPU or do we have to chunk it ourselves?
    - in this case you would have to deal with this in the code manually, copying a chunk of the image, processing it, then copying it back and repeat

## 📚 Resources 

* [Upcoming eScience Center workshops](https://www.esciencecenter.nl/digital-skills/)
* [HackMD Markdown Guide](https://www.markdownguide.org/tools/hackmd/)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [Numba User Guide](https://numba.pydata.org/numba-doc/latest/user/index.html)
* [NL-RSE meetup December 1st](https://www.eventbrite.co.uk/e/nl-rse-meetup-december-1-2021-tickets-195099246097)
