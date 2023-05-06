# wgpu-compute-toy

This is the compute shader engine for https://compute.toys

As well as running on the web via WebAssembly and WebGPU, it can run natively using standard desktop graphics APIs like Vulkan and DirectX.

## Native

```sh
cargo run examples/davidar/buddhabrot.wgsl
```

![screenshot](https://user-images.githubusercontent.com/24291/230871630-7bee3977-8d24-4259-8af6-639232929672.png)

## Web

[Install wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
and run `wasm-pack build`.

The following command can be used to quickly recompile while the [compute.toys](https://github.com/compute-toys/compute.toys) server is running:

```
wasm-pack build --dev && cp -rv pkg/. ../compute.toys/lib/wgputoy/
```
