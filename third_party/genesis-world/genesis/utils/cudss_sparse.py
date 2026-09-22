"""Minimal FP64 SPD cuDSS factor used by the implicit FEM runtime."""

import importlib.metadata as metadata
from pathlib import Path
import time

import numpy as np


def _set_config(cudss, config, param, value):
    array = np.asarray([value], dtype=cudss.get_config_param_dtype(param))
    cudss.config_set(config, param, array.ctypes.data, array.nbytes)


class CudssSpdFactor:
    """Reusable-symbolic lower-CSR cuDSS factor with CPU and CUDA RHS entry points."""

    def __init__(self, matrix):
        import cupy as cp
        from nvmath.bindings import cudss

        matrix = matrix.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int32), np.diff(matrix.indptr))
        self._lower_source = np.flatnonzero(rows >= matrix.indices).astype(np.int32)
        lower_rows = rows[self._lower_source]
        lower_row_ptr = np.empty(matrix.shape[0] + 1, dtype=np.int32)
        lower_row_ptr[0] = 0
        np.cumsum(np.bincount(lower_rows, minlength=matrix.shape[0]), out=lower_row_ptr[1:])
        lower_col_ind = matrix.indices[self._lower_source].astype(np.int32, copy=True)

        self._cp = cp
        self._cudss = cudss
        self._shape = matrix.shape
        self._indptr = matrix.indptr.astype(np.int32, copy=True)
        self._indices = matrix.indices.astype(np.int32, copy=True)
        self._row_ptr = cp.asarray(lower_row_ptr)
        self._col_ind = cp.asarray(lower_col_ind)
        self._values = cp.empty(self._lower_source.size, dtype=cp.float64)

        self._handle = cudss.create()
        self._config = cudss.config_create()
        self._data = cudss.data_create(self._handle)
        self._stream = cp.cuda.get_current_stream()
        cudss.set_stream(self._handle, self._stream.ptr)
        distribution_root = Path(metadata.distribution("nvidia-cudss-cu12").locate_file(""))
        threading_layer = distribution_root / "nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0"
        cudss.set_threading_layer(self._handle, str(threading_layer.resolve()))
        _set_config(cudss, self._config, cudss.ConfigParam.IR_N_STEPS, 0)
        _set_config(cudss, self._config, cudss.ConfigParam.HOST_NTHREADS, 4)
        _set_config(cudss, self._config, cudss.ConfigParam.HYBRID_EXECUTE_MODE, 0)
        self._matrix = cudss.matrix_create_csr(
            matrix.shape[0],
            matrix.shape[1],
            self._values.size,
            self._row_ptr.data.ptr,
            0,
            self._col_ind.data.ptr,
            self._values.data.ptr,
            cudss.DataType.R_32I,
            cudss.DataType.R_32I,
            cudss.DataType.R_64F,
            cudss.MatrixType.SPD,
            cudss.MatrixViewType.LOWER,
            cudss.IndexBase.ZERO,
        )
        analysis_start = time.perf_counter()
        self._execute(cudss.Phase.ANALYSIS, 0, 0)
        self._stream.synchronize()
        self.analysis_seconds = time.perf_counter() - analysis_start
        self.factor_seconds_total = 0.0
        self.solve_seconds_total = 0.0
        self.refactor(matrix)

    def _execute(self, phase, solution_matrix, rhs_matrix):
        self._cudss.execute(
            self._handle,
            phase,
            self._config,
            self._data,
            self._matrix,
            solution_matrix,
            rhs_matrix,
        )

    def refactor(self, matrix):
        matrix = matrix.tocsr(copy=False)
        if not self.matches_pattern(matrix):
            raise RuntimeError("cuDSS implicit-FEM sparsity changed after symbolic analysis")
        start = time.perf_counter()
        self._values.set(np.asarray(matrix.data[self._lower_source], dtype=np.float64))
        self._execute(self._cudss.Phase.FACTORIZATION, 0, 0)
        self._stream.synchronize()
        elapsed = time.perf_counter() - start
        self.factor_seconds_total += elapsed
        return elapsed

    def refactor_lower_values(self, values):
        import torch

        if isinstance(values, torch.Tensor):
            values = self._cp.from_dlpack(values.detach())
        if values.shape != self._values.shape or values.dtype != self._cp.float64:
            raise ValueError("cuDSS lower values have the wrong shape or dtype")
        start = time.perf_counter()
        self._values[...] = values
        self._execute(self._cudss.Phase.FACTORIZATION, 0, 0)
        self._stream.synchronize()
        elapsed = time.perf_counter() - start
        self.factor_seconds_total += elapsed
        return elapsed

    def matches_pattern(self, matrix):
        matrix = matrix.tocsr(copy=False)
        return not (
            matrix.shape != self._shape
            or not np.array_equal(matrix.indptr, self._indptr)
            or not np.array_equal(matrix.indices, self._indices)
        )

    def close(self):
        self._cudss.matrix_destroy(self._matrix)
        self._cudss.data_destroy(self._handle, self._data)
        self._cudss.config_destroy(self._config)
        self._cudss.destroy(self._handle)

    def _solve_cupy(self, rhs):
        cp = self._cp
        cudss = self._cudss
        was_vector = rhs.ndim == 1
        if was_vector:
            rhs = rhs.reshape(self._shape[0], 1)
        rhs = cp.asfortranarray(rhs, dtype=cp.float64)
        solution = cp.empty(rhs.shape, dtype=cp.float64, order="F")
        rhs_matrix = cudss.matrix_create_dn(
            rhs.shape[0],
            rhs.shape[1],
            rhs.shape[0],
            rhs.data.ptr,
            cudss.DataType.R_64F,
            cudss.Layout.COL_MAJOR,
        )
        solution_matrix = cudss.matrix_create_dn(
            solution.shape[0],
            solution.shape[1],
            solution.shape[0],
            solution.data.ptr,
            cudss.DataType.R_64F,
            cudss.Layout.COL_MAJOR,
        )
        start = time.perf_counter()
        self._execute(cudss.Phase.SOLVE, solution_matrix, rhs_matrix)
        self._stream.synchronize()
        self.solve_seconds_total += time.perf_counter() - start
        cudss.matrix_destroy(solution_matrix)
        cudss.matrix_destroy(rhs_matrix)
        return solution[:, 0] if was_vector else solution

    def solve(self, rhs):
        rhs_device = self._cp.asarray(np.asarray(rhs, dtype=np.float64))
        return self._cp.asnumpy(self._solve_cupy(rhs_device))

    def solve_pcg(self, matrix, rhs, *, max_iterations, relative_tolerance):
        """Solve the current SPD matrix with this (possibly older) exact factor as PCG M."""
        import cupyx.scipy.sparse as cpxs

        cp = self._cp
        current = matrix.tocsr(copy=False)
        current_gpu = cpxs.csr_matrix(
            (
                cp.asarray(current.data, dtype=cp.float64),
                cp.asarray(current.indices, dtype=cp.int32),
                cp.asarray(current.indptr, dtype=cp.int32),
            ),
            shape=current.shape,
        )
        b = cp.asarray(np.asarray(rhs, dtype=np.float64))
        b_norm = float(cp.linalg.norm(b).item())
        threshold = relative_tolerance * b_norm
        x = cp.zeros_like(b)
        residual = b.copy()
        residual_norm = b_norm
        iterations = 0
        if residual_norm > threshold:
            z = self._solve_cupy(residual)
            direction = z.copy()
            residual_dot_z = cp.dot(residual, z)
            for iterations in range(1, max_iterations + 1):
                product = current_gpu @ direction
                alpha = residual_dot_z / cp.dot(direction, product)
                x += alpha * direction
                residual -= alpha * product
                residual_norm = float(cp.linalg.norm(residual).item())
                if residual_norm <= threshold:
                    break
                z = self._solve_cupy(residual)
                next_residual_dot_z = cp.dot(residual, z)
                direction = z + (next_residual_dot_z / residual_dot_z) * direction
                residual_dot_z = next_residual_dot_z
        relative_residual = residual_norm / max(b_norm, np.finfo(np.float64).tiny)
        return cp.asnumpy(x), iterations, relative_residual, residual_norm <= threshold

    def solve_gpu(self, rhs):
        import torch

        if not isinstance(rhs, torch.Tensor) or rhs.device.type != "cuda" or rhs.dtype != torch.float64:
            raise ValueError("cuDSS RHS must be a CUDA torch.float64 tensor")
        device_index = rhs.device.index
        with torch.cuda.device(device_index):
            torch_stream = torch.cuda.current_stream(device_index)
            with self._cp.cuda.Device(device_index), self._cp.cuda.ExternalStream(
                torch_stream.cuda_stream, device_id=device_index
            ):
                self._stream = self._cp.cuda.get_current_stream()
                self._cudss.set_stream(self._handle, self._stream.ptr)
                rhs_device = self._cp.from_dlpack(rhs.detach())
                solution = self._solve_cupy(rhs_device)
                return torch.from_dlpack(solution)

    def matvec_gpu(self, rhs):
        import cupyx.scipy.sparse as cpxs
        import torch

        if not isinstance(rhs, torch.Tensor) or rhs.device.type != "cuda" or rhs.dtype != torch.float64:
            raise ValueError("cuDSS matrix-vector input must be a CUDA torch.float64 tensor")
        vector = self._cp.from_dlpack(rhs.detach())
        lower = cpxs.csr_matrix(
            (self._values, self._col_ind, self._row_ptr), shape=self._shape
        )
        diagonal = lower.diagonal()
        result = lower @ vector + lower.T @ vector - diagonal * vector
        return torch.from_dlpack(result)


class CudssSpdMatrix:
    """Current symmetric matrix view backed by a ``CudssSpdFactor`` value buffer."""

    def __init__(self, factor):
        self.factor = factor
        self.shape = factor._shape

    def __matmul__(self, rhs):
        import torch

        rhs_gpu = torch.as_tensor(
            np.asarray(rhs, dtype=np.float64), device="cuda", dtype=torch.float64
        )
        return self.factor.matvec_gpu(rhs_gpu).cpu().numpy()


class CudssDeviceSpdFactor(CudssSpdFactor):
    """One-shot cuDSS factor constructed directly from a CUDA lower CSR matrix."""

    def __init__(self, lower):
        import cupy as cp
        from nvmath.bindings import cudss

        lower = lower.tocsr(copy=True)
        lower.sum_duplicates()
        lower.sort_indices()
        self._cp = cp
        self._cudss = cudss
        self._shape = lower.shape
        self._row_ptr = lower.indptr.astype(cp.int32, copy=False)
        self._col_ind = lower.indices.astype(cp.int32, copy=False)
        self._values = lower.data.astype(cp.float64, copy=False)
        self._handle = cudss.create()
        self._config = cudss.config_create()
        self._data = cudss.data_create(self._handle)
        self._stream = cp.cuda.get_current_stream()
        cudss.set_stream(self._handle, self._stream.ptr)
        distribution_root = Path(metadata.distribution("nvidia-cudss-cu12").locate_file(""))
        threading_layer = distribution_root / "nvidia/cu12/lib/libcudss_mtlayer_gomp.so.0"
        cudss.set_threading_layer(self._handle, str(threading_layer.resolve()))
        _set_config(cudss, self._config, cudss.ConfigParam.IR_N_STEPS, 0)
        _set_config(cudss, self._config, cudss.ConfigParam.HOST_NTHREADS, 4)
        _set_config(cudss, self._config, cudss.ConfigParam.HYBRID_EXECUTE_MODE, 0)
        self._matrix = cudss.matrix_create_csr(
            lower.shape[0],
            lower.shape[1],
            self._values.size,
            self._row_ptr.data.ptr,
            0,
            self._col_ind.data.ptr,
            self._values.data.ptr,
            cudss.DataType.R_32I,
            cudss.DataType.R_32I,
            cudss.DataType.R_64F,
            cudss.MatrixType.SPD,
            cudss.MatrixViewType.LOWER,
            cudss.IndexBase.ZERO,
        )
        analysis_start = time.perf_counter()
        self._execute(cudss.Phase.ANALYSIS, 0, 0)
        self._stream.synchronize()
        self.analysis_seconds = time.perf_counter() - analysis_start
        factor_start = time.perf_counter()
        self._execute(cudss.Phase.FACTORIZATION, 0, 0)
        self._stream.synchronize()
        self.factor_seconds_total = time.perf_counter() - factor_start
        self.solve_seconds_total = 0.0
