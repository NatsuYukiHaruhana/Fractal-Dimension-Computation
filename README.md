<h1 align="center">Fractal Dimension Computation</h1>
<h2 align="center">🔎About🔎</h2>
<p>This work presents a framework for testing the runtime and results of fractal dimension computation algorithms. Included are two such algorithms by Juan Ruiz de Miras and Miguel Ángel Posadas. The first box-counting algorithm can be found <a href="https://www.ugr.es/~demiras/fbc/" target="_blank">here</a>. The differential box-counting algorithm does not have a proper source code link from the authors. They have been re-written and edited as described in their paper, which can be found <a href="https://www.researchgate.net/publication/336605788_Fast_differential_box-counting_algorithm_on_GPU" target="_blank">here</a>. A Jupyter Notebook file through which you can generate input data and run the built executables of the algorithms is also attached.</p>

###

<br clear="both">

---
<h2 align="center">✅Features✅</h2>

- sequential Fast Box Counting algorithm running on CPU;
- parallelized Fast Box Counting algorithm running on GPU;
- sequential Differential Box Counting algorithm running on CPU;
- parallelized Differential Box Counting algorithm running on GPU;
- Jupyter Notebook Wrapper file used for image generation and test results.

###

<br clear="both">

---
<h2 align="center">⚙️Installation⚙️</h2>
<p>CMake may be used to build both the CPU and GPU versions of the algorithms. Due to the GPU versions of the algorithms specifically using the CUDA framework, which is only available on certain nVidia GPUs, we have two separate make commands.</p>
<p>They each generate one directory and each generate an executable with the name FractalDimensionComputation, though there is a distinction for the GPU executable having "_GPU" appended at the end of the file name.</p>
<p>If, for whatever reason, you would like to delete the build directories, you can do so with the</p>

```bash
make clean
```

<p>command. Please keep in mind that the deletion of these directories is afterwards immediate and cannot be reverted.</p>

<h3 align="center">👩‍💻CPU Versions of the algorithms👩‍💻</h3>
<p>The CPU versions of the algorithms require OpenCV for image reading and edge detection purposes.</p>

```bash
make build_cpu
```

<h3 align="center">👩‍💻GPU Versions of the algorithms👩‍💻</h3>
<p>The GPU versions of the algorithms require OpenCV for image reading and edge detection purposes, as well as the CUDA framework for allowing the CUDA kernels to properly compile.</p>

```bash
make build_gpu
```

<h3 align="center">📔Jupyter Notebook📔</h3>
<p>Whichever IDE of your choosing which is able to run Jupyter Notebooks will suffice. The requirements are listed within the Notebook. Simply run the first cell for them to be installed. The code was tested only on Visual Studio Code and Python version 3.13.0.</p>

###

<br clear="both">

---
<h2 align="center">💻Usage💻</h2>

<p>⚠️Depending on the task you wish to achieve, you must first keep in mind to have the relevant executable file in the same working directory as your Jupyter Notebook file.⚠️</p>

<h3 align="center">🖼️Image Generation🖼️</h3>

<p>⚠️The program takes a screenshot of your current screen. Placing your cursor in front of the window could lead to unwanted alterations in fractal dimension computations. If your Desktop size is anything other than 100%, the screenshot will NOT be taken correctly. In this case, please wait for the program to finish drawing the fractal (usually indicated by all arrows disappearing), and take a screenshot yourself (for example, on Windows 10 and 11 you can press WIN + SHIFT + S to take a screenshot).⚠️</p>

<p>The Notebook is capable of generating four different kinds of fractals:</p>

- Julia Sets
- Sierpinski Triangles
- Sierpinski Pentagons
- Koch Curves

<p>To generate one such image, the PIL, tkinter and turtle packages are used for screenshots, window generation, respectively drawing the fractal itself.</p>
<p>Simply run the imports cell, the code cell under "Draw Image", and lastly the cells under the image which you wish to generate. You may modify any relevant parameters, such as cX and cY for your Julia set</p>

<h3 align="center">🧪Algorithm Testing and Benchmarking🧪</h3>

<p>⚠️As to not clutter your working directory, the algorithms look for the following folders in your working directory, so please make sure you have them:⚠️</p>

- images (must be explicitly created by you)
- {filename} (created by the Jupyter Notebook if it doesn't exist)

<p>Below are the arguments given to the executable, and what you should be setting them to according to your wants and needs:</p>

```python
runTimes = 10 # How many times should the algorithm run? If you don't wish to benchmark the running times, you should leave this as "1", given that the resulting dimension does not change between runs over the same file. One "results" file will be generated for each run.
filename = "Triangle" # This should be the same name as the image which you wish to use. Do NOT add the extension, though keep in mind it HAS to be a .png file. A directory will be created with the name of this string if it hasn't already been created.
algorithm = "DBC" # Value can be either "BC" for the Fast Box Counting algorithm, or "DBC" for the Differential Box Counting algorithm.
runOn = "CPU" # This helps make benchmarking easier by allowing you to specify the ending of the executable name. For example, if you rename your CPU build of the program to "Fast Box Counting CPU" and your GPU build of the program to "Fast Box Counting GPU", runOn may be either "CPU" or "GPU". This is not actually an argument used when calling the executable, merely a way to differentiate between the two executables.
```

<p>If everything runs correctly, you should get a return value of 0. Otherwise, there is a problem with your executable or with how you are calling the executable. If the problem lies in your argument usage, a text file called "errors.txt" will be generated in your working directory, detailing all errors which lead to the premature stop of the executable.</p>
<p>⚠️In the case that you have already run one of your executables on an image, the results from the previous run(s) WILL be replaced.⚠️</p>

###

<br clear="both">

---
<h2 align="center">📋Work History📋</h2>
Week 9 - Created README.md and LICENSE files.
Week 12 - Added DBC (differential box-counting) algorithm.
Week 13 - Added enhancements to the Jupyter Notebook Wrapper file.
Week 14 - Updated README.md and pushed changes on GitHub.
###

<br clear="both">

---
<h2 align="center">📜License📜</h2>
The project is licensed under [the MIT license.](./LICENSE).

The Fast Box Counting algorithm has been licensed under CC-BY-SA-4.0 by Juan Ruiz de Miras and Miguel Ángel Posadas. The original code can be found [here](https://www.ugr.es/~demiras/fbc/). Their code may be found within the following files:
- bcCPU.cpp
- bcCPU.h
- main.cpp
- bcCUDA2D.cu
- bcCUDA2D.cuh
- mainGPU.cpp
