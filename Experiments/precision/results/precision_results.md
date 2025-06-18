# Precision results for the interaction block

## On CPU

This has been ran with no CuEq and conv_tp

| Precision | Abs Diff Avg (Mean ± Std) | Abs Diff Avg Max | Rel Diff Avg (Mean ± Std) | Rel Diff Avg Max | Comments                                                                                                                                                             |
| --------- | ------------------------- | ---------------- | ------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FP32**  | 3.85e-09 ± 3.77e-09       | 4.88e-08         | 8.40e-06 ± 2.17e-03       | 9.04e-01         | FP32 is extremely close to FP64: errors are near machine epsilon, with a high relative diff max likely caused by division by small values. Suitable for CPU use.     |
| **FP16**  | 1.12e-05 ± 1.09e-05       | 1.28e-04         | 1.71e-02 ± 2.36e+00       | 9.41e+02         | FP16 introduces visible absolute errors and large relative diff max due to small FP64 reference values. No hardware acceleration means no FP16 speed benefit on CPU. |


## On GPU

This has been ran with no CuEq and conv_tp

| Precision | Abs Diff Avg (Mean ± Std) | Abs Diff Avg Max | Rel Diff Avg (Mean ± Std) | Rel Diff Avg Max | Comments                                                                                                                                                          |
| --------- | ------------------------- | ---------------- | ------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FP32**  | 3.24e-09 ± 3.22e-09       | 4.50e-08         | 5.63e-06 ± 1.09e-03       | 4.50e-01         | FP32 is extremely close to FP64 reference; small absolute and relative differences. GPU ops are precise, consistent with CPU trends.                              |
| **FP16**  | 1.21e-05 ± 1.21e-05       | 1.47e-04         | 1.23e-02 ± 7.07e-01       | 1.83e+02         | FP16 shows expected larger differences, especially in relative terms where FP64 values are small. GPU’s FP16 performance is reasonable for speed-sensitive tasks. |
