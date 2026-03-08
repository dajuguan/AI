use tch::Tensor;

/// Calculate the memory usage of a tensor in bytes.
pub fn get_mem_usage(x: Tensor) -> u64 {
    let numel = x.numel() as u64;
    let bytes_per_element = x.kind().elt_size_in_bytes() as u64;
    numel.saturating_mul(bytes_per_element)
}

#[cfg(test)]
mod tests {
    use tch::Kind;
    use tch::kind::FLOAT_CPU;

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
        } else {
            assert_eq!(t.device(), tch::Device::Cpu);
        }
    }
}
