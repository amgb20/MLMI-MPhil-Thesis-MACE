# Comparison Results on Numerical Stability

## Study Objective

This preliminary study investigates the impact of numerical precision on the computational performance of MACE models, with a specific focus on the interaction block - the most computationally expensive component responsible for approximately 80% of the total computation during both training and inference.

The primary goal is to determine whether reducing numerical precision can significantly accelerate MACE computations while maintaining acceptable accuracy. This investigation is motivated by the desire to optimize MACE's computational efficiency without compromising model performance.

## Technical Focus

Our analysis centers on the `conv_fusion` step within the interaction block, which involves complex tensor operations and represents the computational bottleneck in MACE architectures. When the cuEquivariance CUDA kernel is activated, tensor calculations are optimized through a loop that relies on the [SegmentedPolynomial](https://docs.nvidia.com/cuda/cuequivariance/api/generated/cuequivariance_torch.SegmentedPolynomial.html) class.

This class is constrained to accept only float64/32 precision internally, but the surrounding input/output tensors can be expressed in lower precision formats. Therefore, we are systematically testing the following precision configurations:

- **fp64** (double precision)
- **fp32** (single precision) 
- **tf32** (TensorFloat-32)
- **bf16** (Brain Float 16)
- **fp16** (half precision)

## Benchmark Strategy

Our evaluation uses fp64 as the baseline benchmark, reflecting the standard practice in most MACE model training scenarios. All precision configurations will be compared against this fp64 reference to assess both computational speedup and potential accuracy trade-offs.

## Experimental Parameters

To evaluate performance under extreme computational conditions, we conducted our benchmarks using large-scale graph structures:
- **Number of Nodes**: 100
- **Number of Edges**: 10,000

These parameters were chosen to stress-test the numerical precision implementations under realistic high-complexity scenarios.

## Hardware Configuration

- precision test were conducted on a NVIDIA GeForce RTX 2080 Ti
- GPU and Wall-time on a NVIDIA A100-SXM4-80GB

## GPU Memory and Computational Performance

The following tables present GPU memory allocation and computational timing results for different precision configurations, with fp64 serving as the baseline reference.

A warm-up is needed for ...

### Results with math_dtype = fp64

| Precision | L0 Time (ms) | L0 Memory (MB) | L1 Time (ms) | L1 Memory (MB) | L1/L0 Time Ratio | L1/L0 Memory Ratio |
|-----------|--------------|----------------|--------------|----------------|-------------------|-------------------|
| **FP64**  | 0.583        | 384.42         | 1.827        | 386.853        | 3.134             | 1.006             |
| FP32      | 0.582        | 384.42         | 2.060        | 386.853        | 3.540             | 1.006             |
| TF32      | 0.579        | 384.42         | 1.816        | 386.853        | 3.136             | 1.006             |
| FP16      | 0.480        | 384.42         | 1.588        | 386.853        | 3.308             | 1.006             |
| BF16      | 0.485        | 384.42         | 1.621        | 386.853        | 3.342             | 1.006             |

![alt text](../figs/mace_layer_bm_fp64_gpu_time.png)

### Results with math_dtype = fp32

| Precision | L0 Time (ms) | L0 Memory (MB) | L1 Time (ms) | L1 Memory (MB) | L1/L0 Time Ratio | L1/L0 Memory Ratio |
|-----------|--------------|----------------|--------------|----------------|-------------------|-------------------|
| FP64      | 0.310        | 383.576        | 0.553        | 384.805        | 1.784             | 1.003             |
| **FP32**  | 0.304        | 383.576        | 0.527        | 384.805        | 1.734             | 1.003             |
| TF32      | 0.300        | 383.576        | 0.527        | 384.805        | 1.757             | 1.003             |
| FP16      | 0.299        | 383.576        | 0.532        | 384.805        | 1.779             | 1.003             |
| BF16      | 0.298        | 383.576        | 0.527        | 384.805        | 1.768             | 1.003             |

![alt text](../figs/mace_layer_bm_fp32_gpu_time.png)

### Key Observations

- **Memory Usage**: TBD
- **Computational Performance**: TBD
- **Layer Scaling**: TBD

### Wall-time Performance Analysis

The wall-time measurements quantify the actual computational time required for tensor operations across different precision configurations, providing insights into real-world performance characteristics.


## Numerical Precision Benchmark Analysis

Our numerical precision benchmark evaluates the accuracy trade-offs when reducing precision from the baseline fp64 configuration. We compare each precision format against fp64 by calculating:

- **Maximum Absolute Error**: The largest absolute difference between fp64 and lower precision results
- **Maximum Relative Error**: The largest relative difference, indicating proportional accuracy loss

### Experimental Setup

We perform both **forward** and **backward** passes for each layer of the interaction blocks using a simple quadratic loss function:
```python
def loss_fn(out):
    return out.pow(2).sum()
```

### Sparse Graph Results (math_dtype = fp64)

The sparse graph configuration uses edge dimensions of (2, E), where E represents the number of edges.

JIT IMPLEMENTATION everywhere above

#### Layer 0 Precision Comparison

| Pass Type | Precision | Max Absolute Error | Max Relative Error |
|-----------|-----------|-------------------|-------------------|
| **Forward** | FP32 | 0.000004 | 0.094181 |
| **Forward** | TF32 | 0.000004 | 0.094181 |
| **Forward** | FP16 | 0.032195 | 3438.831517 |
| **Forward** | BF16 | 0.312769 | 13234.022459 |
| **Backward** | FP32 | 0.000385 | 0.035966 |
| **Backward** | TF32 | 0.000385 | 0.035966 |
| **Backward** | FP16 | 3.275100 | 266.350737 |
| **Backward** | BF16 | 21.635856 | 778.698955 |

#### Layer 1 Precision Comparison

| Pass Type | Precision | Max Absolute Error | Max Relative Error |
|-----------|-----------|-------------------|-------------------|
| **Forward** | FP32 | 0.000004 | 0.118674 |
| **Forward** | TF32 | 0.000004 | 0.118674 |
| **Forward** | FP16 | 0.025441 | 1127.853786 |
| **Forward** | BF16 | 0.209613 | 10074.333409 |
| **Backward** | FP32 | 0.000363 | 0.455505 |
| **Backward** | TF32 | 0.000363 | 0.455505 |
| **Backward** | FP16 | 2.917152 | 2957.682628 |
| **Backward** | BF16 | 25.295468 | 15769.572075 |

### Key Observations

- **FP32/TF32 Performance**: TBD
- **BF16 Limitations**: TBD
- **Layer Consistency**: TBD


![alt text](../figs/precision_fp64.png)
![alt text](../figs/precision_fp64_1.png)


### Building a fully connected graph

That way the number of edges dim is (2, N^2)

all the plots are loglog node vs mae

![alt text](../figs/nodevsmae.png)

loglog edge vs maer

![alt text](../figs/edgevsmae.png)