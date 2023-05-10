# wgpu-compute-toy

This is the compute shader engine for https://compute.toys

As well as running on the web via WebAssembly and WebGPU, it can run natively using standard desktop graphics APIs like Vulkan and DirectX.

## Native

```sh
cargo run examples/davidar/buddhabrot.wgsl
```

![screenshot](https://user-images.githubusercontent.com/24291/230871630-7bee3977-8d24-4259-8af6-639232929672.png)

## Web

See https://github.com/compute-toys/compute.toys
