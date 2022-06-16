use crate::WGSLError;

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
    .or(Err(WGSLError {
        summary: format!("Cannot parse '{value}' as u32"),
        line,
    }))
}
