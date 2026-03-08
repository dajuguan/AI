use tch::Tensor;

/// Calculate the memory usage of a tensor in bytes.
pub fn get_mem_usage(x: Tensor) -> u64 {
    let numel = x.numel() as u64;
    let bytes_per_element = x.kind().elt_size_in_bytes() as u64;
    numel.saturating_mul(bytes_per_element)
}

#[cfg(test)]
mod tests {
    use tch::kind::FLOAT_CPU;

    use super::*;

    #[test]
    fn test_get_mem_usage() {
        let t = Tensor::rand([2, 3], FLOAT_CPU);
        // float32 is 4 bytes
        let expected = 4 * 6;
        assert_eq!(get_mem_usage(t), expected);
    }
}
