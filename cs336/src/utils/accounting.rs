use std::hint::black_box;
use tch::Tensor;

use crate::utils::{CUDA_DEVICE, DEVICE_INDEX};

/// Calculate the memory usage of a tensor in bytes.
pub fn get_mem_usage(x: Tensor) -> u64 {
    let numel = x.numel() as u64;
    let bytes_per_element = x.kind().elt_size_in_bytes() as u64;
    numel.saturating_mul(bytes_per_element)
}

pub fn time_matmul(a: &Tensor, b: &Tensor) -> f64 {
    if tch::Cuda::is_available() {
        tch::Cuda::synchronize(DEVICE_INDEX);
    }
    let a = a.to(CUDA_DEVICE);
    let b = b.to(CUDA_DEVICE);

    let num_trials = 20;
    let start = std::time::Instant::now();
    for _ in 0..num_trials {
        let y = a.matmul(&b);
        black_box(&y); // incase compiler optimizes away the matmul
    }

    if tch::Cuda::is_available() {
        tch::Cuda::synchronize(DEVICE_INDEX);
    }

    let duration = start.elapsed();
    println!("Total Time taken for matmul {:.6?} seconds", duration);
    (duration / num_trials).as_secs_f64()
}

/// Calculate float point opeations
pub fn flops(tokens: u64, params: u64) -> u64 {
    return 2 * tokens * params;
}

#[cfg(test)]
mod tests {
    use tch::Kind;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};

    use super::*;

    #[test]
    fn test_get_mem_usage() {
        let t = Tensor::rand([2, 3], FLOAT_CPU);
        // float32 is 4 bytes
        let expected = 4 * 6;
        assert_eq!(get_mem_usage(t), expected);
    }

    #[test]
    fn test_b16_info() {
        let x = Tensor::from_slice(&[1e-8_f32]);

        let x_f16 = x.to_kind(Kind::Half).to_kind(Kind::Float);
        let x_bf16 = x.to_kind(Kind::BFloat16).to_kind(Kind::Float);

        let v_f32 = x.double_value(&[0]);
        let v_f16 = x_f16.double_value(&[0]);
        let v_bf16 = x_bf16.double_value(&[0]);

        println!("f32  = {:.20e}", v_f32);
        println!("f16  = {:.20e}", v_f16);
        println!("bf16 = {:.20e}", v_bf16);

        assert_eq!(v_f16, 0.0, "1e-8 should underflow to zero in float16");
        assert_ne!(v_bf16, 0.0, "1e-8 should stay non-zero in bfloat16");
    }

    #[test]
    fn test_device_info() {
        let t = Tensor::from_slice(&[1e-8_f32]);
        let is_available = tch::Cuda::is_available();
        let device_count = tch::Cuda::device_count();
        let cuda_probe_ok = t.f_to(tch::Device::Cuda(0)).is_ok();

        println!("Device: {:?}", t.device());
        println!("Cuda::is_available(): {is_available}");
        println!("Cuda::device_count(): {device_count}");
        println!("CUDA allocation probe: {cuda_probe_ok}");

        if cuda_probe_ok {
            let t_cuda = t.to_device(tch::Device::Cuda(0));
            println!("CUDA Device: {:?}", t_cuda.device());
            assert_eq!(t_cuda.device(), tch::Device::Cuda(0));
            println!("tch::utils::has_cuda(): {}", tch::utils::has_cuda());
            println!("tch::utils::has_cudart(): {}", tch::utils::has_cudart());
            println!(
                "tch::Cuda::cudnn_is_available(): {}",
                tch::Cuda::cudnn_is_available()
            );
            println!(
                "tch::utils::version_cudnn(): {}",
                tch::utils::version_cudnn()
            );
            println!(
                "tch::utils::version_cudart(): {}",
                tch::utils::version_cudart()
            );
        } else {
            assert_eq!(t.device(), tch::Device::Cpu);
        }
    }

    #[test]
    fn test_flops() {
        let B = 16384; // number of points
        let D = 32768; // dimension of each point
        let K = 8192; // number of outputs

        let x = Tensor::rand([B, D], FLOAT_CUDA);
        let w = Tensor::rand([D, K], FLOAT_CUDA);

        let params = D * K;
        let actual_flops = flops(B as u64, params as u64);
        let duration = time_matmul(&x, &w);
        let actual_flop_per_sec = actual_flops as f64 / duration;
        println!("Time taken per matmul {:.6} seconds", duration);
        println!("FLOPS: {:.2} TFLOPS", actual_flop_per_sec / 1e12);

        // results for 5090 Ti
        // Actural FLOPS: 17.07 TFLOPS
        // Promised FLOPS: 29.15 TFLOPS
    }
}
