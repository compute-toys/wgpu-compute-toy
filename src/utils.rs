use crate::WGSLError;
use cached::proc_macro::cached;
use gloo_net::http::Request;
use js_sys::Promise;
use std::future::Future;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

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

#[cached]
pub async fn fetch_include(name: String) -> Option<String> {
    let url = format!("https://compute-toys.github.io/include/{name}.wgsl");
    let resp = Request::get(&url).send().await.ok()?;
    if resp.status() == 200 {
        resp.text().await.ok()
    } else {
        None
    }
}

pub fn promise<F, T>(future: F) -> Promise
where
    F: Future<Output = Option<T>> + 'static,
    JsValue: From<T>,
{
    let mut future = Some(future);

    Promise::new(&mut |resolve, _reject| {
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
