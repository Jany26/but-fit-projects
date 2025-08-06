# BUT FIT Projects
A collection of interesting small projects completed as part of various courses during studies at BUT FIT.

Here is a mapping of each project (directory) to their topic.

| Project name                               | Topic                         |
|--------------------------------------------|-------------------------------|
|`binary-code-analysis`                      | Low-level Projects            |
|`bitvector-steganography`                   | Low-level Projects            |
|`car-accidents-analysis`                    | Data Mining                   |
|`diabetic-data-analysis`                    | Data Mining                   |
|`directory-file-histogram`                  | Low-level Projects            |
|`geometric-variability-of-proteins`         | Biological Data Analysis      |
|`hash-table-implementation`                 | Low-level Projects            |
|`mandelbrot-vector-openmp`                  | Parallel Computation Projects |
|`marching-cubes-parallel-openmp`            | Parallel Computation Projects |
|`metagenomics`                              | Biological Data Analysis      |
|`mutation-effect-on-proteins-prediction`    | Biological Data Analysis      |
|`ngs-intron-detection`                      | Biological Data Analysis      |
|`no-sql-demonstration`                      | Data Mining                   |
|`parallel-game-of-life`                     | Parallel Computation Projects |
|`particle-interactions-cuda`                | Parallel Computation Projects |
|`particle-interactions-openacc`             | Parallel Computation Projects |
|`pipeline-merge-sort`                       | Parallel Computation Projects |
|`pymol-correlated-mutations`                | Biological Data Analysis      |
|`semaphores`                                | Low-level Projects            |
|`triangular-maze-solver`                    | Low-level Projects            |

What follows is a brief description of each project grouped by topic, related courses and used technologies.

## Biological Data Analysis

Related Courses:
- [Bioinformatics](https://www.fit.vut.cz/study/course/BIF)
- [Advanced Bioinformatics](https://www.fit.vut.cz/study/course/PBI)

Used Technologies: PyMOL, Python, BioPython, Bash/Shell, other Bioinformatics tools (Megan, Diamond, igv, ...)

`geometric-variability-of-proteins`
- Measure the variability of covalent-bond angles at alpha carbon positions along the protein spine and visually represent the variability in PyMOL.

`metagenomics`
- Analyse the metagenomic content of a synthetically prepared randomised sample of RNAs from Salmonella and E.Coli organisms from NCBI database.

`mutation-effect-on-proteins-prediction`
- Knowledge-based predictor of the mutation effects on the protein's function using evolution information and physical/chemical properties.

`ngs-intron-detection`
- Estimate gene expression of individual members of a specific gene cluster, while considering the presence of introns.

`pymol-correlated-mutations`
- From a family of similar and related protein sequences, using mutual information, find the most correlated mutations and visualize them in PyMOL.

## Data Mining Projects

Related Courses:

- [Data Analysis and Visualization in Python](https://www.fit.vut.cz/study/course/IZV/)
- [Data Storage and Preparation](https://www.fit.vut.cz/study/course/UPA/)
- [Knowledge Discovery in Databases](https://www.fit.vut.cz/study/course/ZZN/)

Used Technologies: Python, Numpy, Pandas, MongoDB, Neo4j, Docker, Altair AI Studio (formerly RapidMiner), Machine Learning

`car-accidents-analysis`
- Data preparation and statistical analysis of car accident statistics in Czechia in Python.

`diabetic-data-analysis`
- Simple data mining tasks from statistics about patients with diabetes from USA. Preparation and analysis performed using RapidMiner.

`no-sql-demonstration`
- Python scripts demonstrating simple setup, usage and data queries in NoSQL technologies, such as MongoDB and Neo4j.

## Low-level Projects (written mostly in C)

Related Courses:

- [Binary Code Analysis](https://www.fit.vut.cz/study/course/IAN/)
- [The C Programming Language](https://www.fit.vut.cz/study/course/IJC/)
- [Operating Systems](https://www.fit.vut.cz/study/course/IOS/)
- [Introduction to Programming Systems](https://www.fit.vut.cz/study/course/IZP/)

Used Technologies: C, Shell, UNIX knowledge

`binary-code-analysis`
- Simplified version of `readelf -l`, printing ELF program headers (segments) and listing the sections that belong to each segment. Uses the `libelf` and `gelf` APIs to parse and inspect ELF binaries, displaying segment types and permissions.
- Reports analyzing a Linux kernel panic using the `crash` utility and a captured `vmcore` dump.

`bitvector-steganography`
- Memory-efficient bit array library in C using macros and optional inline functions.
- Steganographic message decoding from PPM images.

`directory-file-histogram`
- Shell script that recursively analyzes a directory tree, counts files and directories, and generates a file size histogram (with optional normalized ASCII bar chart), ignoring files matching an optional regex. Supports human-readable output and handles unreadable files gracefully.

`custom-protocol-dissector`
- Dissector of a custom protocol in Lua, able to be plugged into WireShark, along with replicated client implementation based on the given server/client binaries.

`simple-python-remote-downloader`
- File downloader using TCP from a remote server, establishing server address using UDP, written in Python.

`hash-table-implementation`
- Implementation of the UNIX `tail` utility in C.
- Program that counts word frequencies in the input text using a custom modular hash table library (`libhtab`) provided as both a static and shared version.

`semaphores`
- Solution to the modification of a concurrency problem [The Faneuil Hall problem](https://greenteapress.com/semaphores/LittleBookOfSemaphores.pdf) using semaphores and shared memory.

`triangular-maze-solver`
- Simple triangular grid maze solver.

## Parallel Computation Projects

Related Courses:

- [Computation Systems Architectures](https://www.fit.vut.cz/study/course/AVS/)
- [Parallel Computations on GPU](https://www.fit.vut.cz/study/course/PCG/)
- [Parallel and Distributed Algorithms](https://www.fit.vut.cz/study/course/PRL/)

Used Technologies: C/C++, OpenMP, MPI, CUDA, OpenACC

`mandelbrot-vector-openmp`
- Mandelbrot set computation and visualization using vectorization in OpenMP.

`marching-cubes-parallel-openmp`
- 3D rendering using marching cubes algorithm implemented with parallelization in OpenMP.

`parallel-game-of-life`
- Parallel simulation of Game of Life using inter-process communication with MPI (Message Passing Interface).

`particle-interactions-cuda`
- Efficient 3D particle movement, gravity and collision simulation implemented using CUDA.

`particle-interactions-openacc`
- 3D particle movement, gravity and collision simulation implemented using OpenACC.

`pipeline-merge-sort`
- Pipeline Merge Sort implementation using MPI (Message Passing Interface).
