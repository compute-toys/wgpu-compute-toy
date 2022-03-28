# wgputoy

## Build with `wasm-pack build`

```
wasm-pack build
```

## Test in Headless Browsers with `wasm-pack test`

```
wasm-pack test --headless --firefox
```

## Publish to NPM with `wasm-pack publish`

```
wasm-pack publish
```

## Batteries Included

* [`wasm-bindgen`](https://github.com/rustwasm/wasm-bindgen) for communicating
  between WebAssembly and JavaScript.
* [`console_error_panic_hook`](https://github.com/rustwasm/console_error_panic_hook)
  for logging panic messages to the developer console.
* [`wee_alloc`](https://github.com/rustwasm/wee_alloc), an allocator optimized
  for small code size.
