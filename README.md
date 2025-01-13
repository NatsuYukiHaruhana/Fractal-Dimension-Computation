<h1 align="center">Fractal Dimension Computation</h1>
<h2 align="center">🔎About🔎</h2>
<p>This work presents a framework for testing the runtime and results of fractal dimension computation algorithms. Included are two such algorithms by Juan Ruiz de Miras and Miguel Ángel Posadas. The first box-counting algorithm can be found <a href="https://www.ugr.es/~demiras/fbc/" target="_blank">here</a>. The differential box-counting algorithm does not have a proper source code link from the authors. They have been re-written and edited as described in their paper, which can be found <a href="https://www.researchgate.net/publication/336605788_Fast_differential_box-counting_algorithm_on_GPU" target="_blank">here</a>. A Jupyter Notebook file through which you can generate input data and run the built executables of the algorithms is also attached.</p>

###

<br clear="both">

---
<h2 align="center">✅Features✅</h2>

###

<br clear="both">

---
<h2 align="center">⚙️Installation⚙️</h2>
<h3 align="center">CPU Versions of the algorithms</h3>
```bash
make build_cpu
```
<h3 align="center">GPU Versions of the algorithms</h3>
```
make build_gpu
```
<h3 align="center">Jupyter Notebook</h3>

###

<br clear="both">

---
<h2 align="center">💻Usage💻</h2>

###

<br clear="both">

---
<h2 align="center">📋Work History📋</h2>
Week 9 - Created README.md and LICENSE files.

###

<br clear="both">

---
<h2 align="center">📜License📜</h2>
The project is licensed under [the MIT license.](./LICENSE).

The Fast Box Counting algorithm has been licensed under CC-BY-SA-4.0 by Juan Ruiz de Miras and Miguel Ángel Posadas. The original code can be found [here](https://www.ugr.es/~demiras/fbc/). Their code may be found, both modified and unmodified, within the following files:
- bcCPU.cpp
- bcCPU.h
- main.cpp
- bcCUDA2D.cu
- bcCUDA2D.cuh
- mainGPU.cpp
