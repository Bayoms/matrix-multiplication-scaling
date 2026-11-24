# Matrix multiplication scaling in Python

A from-scratch implementation of square matrix multiplication in pure Python, timed at different sizes to see how the cost grows, then checked and compared against NumPy.

## What the notebook does
1. Builds two n × n matrices of random integers (0–9).
2. Multiplies them with the classic triple loop: C[i][j] = Σₖ A[i][k] · B[k][j].
3. Times the run for n = 500 and n = 1000.
4. Verifies the result against NumPy (np.array_equal → True).
5. Times NumPy on the same 1000 × 1000 matrices.

## Results (Google Colab, standard CPU runtime)

| Method | Size | Time |
|---|---|---|
| Pure Python triple loop | 500 × 500 | 13.78 s |
| Pure Python triple loop | 1000 × 1000 | 117.68 s |
| NumPy (@, incl. list→array conversion) | 1000 × 1000 | 2.70 s |

## What the results show
- *O(n³) scaling.* Doubling n means 8× more multiply-add operations (n³). The measured time grew about 8.5×, close to the prediction. The extra cost is likely from memory access, because the inner loop reads B[k][j] down a column, which uses the cache poorly.
- *Extrapolation.* At this rate, 5000 × 5000 would take roughly 4 hours in pure Python.
- *Why NumPy is ~44× faster.* Python interprets every multiply-add one by one as a general object. NumPy runs the loop in compiled C on contiguous arrays. Most of NumPy's 2.7 s here is actually converting the Python lists into arrays.

## Possible improvements
- Reorder the loops (i–k–j) so the inner loop walks along rows and uses the cache better.
- Block/tile the matrices so sub-blocks stay in cache.
- Use float arrays with optimised BLAS, or move to GPU/HPC for very large n.
  
## Run it
Click *Open in Colab* at the top of the notebook and choose Runtime → Run all.
