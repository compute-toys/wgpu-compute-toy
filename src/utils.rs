use crate::WGSLError;

#[cfg(target_arch = "wasm32")]
use {cached::proc_macro::cached, std::future::Future, wasm_bindgen::prelude::*};

pub fn set_panic_hook() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

pub fn parse_u32(value: &str, line: usize) -> Result<u32, WGSLError> {
    let value = value.trim().trim_end_matches('u');
    if value.starts_with("0x") {
        <u32>::from_str_radix(value.strip_prefix("0x").unwrap(), 16)
    } else {
        value.parse::<u32>()
    }
    .or(Err(WGSLError::new(
        format!("Cannot parse '{value}' as u32"),
        line,
    )))
}

#[cfg(target_arch = "wasm32")]
#[cached]
pub async fn fetch_include(name: String) -> Option<String> {
    let url = format!("https://compute-toys.github.io/include/{name}.wgsl");

    #[cfg(target_arch = "wasm32")]
    let resp = gloo_net::http::Request::get(&url).send().await.ok()?;
    #[cfg(not(target_arch = "wasm32"))]
    let resp = reqwest::get(&url).await.ok()?;

    if resp.status() == 200 {
        resp.text().await.ok()
    } else {
        None
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn fetch_include(name: String) -> Option<String> {
    let filename = format!("./include/{name}.wgsl");
    std::fs::read_to_string(filename).ok()
}

#[cfg(target_arch = "wasm32")]
pub fn promise<F, T>(future: F) -> js_sys::Promise
where
    F: Future<Output = Option<T>> + 'static,
    JsValue: From<T>,
{
    let mut future = Some(future);

    js_sys::Promise::new(&mut |resolve, _reject| {
        let future = future.take().unwrap_throw();

        wasm_bindgen_futures::spawn_local(async move {
            let val = if let Some(val) = future.await {
                JsValue::from(val)
            } else {
                JsValue::undefined()
            };
            resolve.call1(&JsValue::undefined(), &val).unwrap_throw();
        });
    })
}
