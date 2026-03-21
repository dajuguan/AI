mod accounting;
mod tensor_basic;

const DEVICE_INDEX: i64 = 0;
const CUDA_DEVICE: tch::Device = tch::Device::Cuda(DEVICE_INDEX as usize);
