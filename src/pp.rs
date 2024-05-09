use crate::{
    bind::NUM_ASSERT_COUNTERS,
    utils::{fetch_include, parse_u32},
};
use async_recursion::async_recursion;
use itertools::Itertools;
use lazy_regex::*;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn wgsl_error_handler(summary: &str, row: usize, col: usize);
}

pub struct WGSLError {
    summary: String,
    line: usize,
}

impl WGSLError {
    pub fn new(summary: String, line: usize) -> Self {
        Self { summary, line }
    }
    pub fn handler(summary: &str, row: usize, col: usize) {
        #[cfg(target_arch = "wasm32")]
        wgsl_error_handler(summary, row, col);
        #[cfg(not(target_arch = "wasm32"))]
        panic!("{}:{}: {}", row, col, summary);
    }
    pub fn submit(&self) {
        Self::handler(&self.summary, self.line, 0)
    }
}

#[wasm_bindgen]
pub struct SourceMap {
    #[wasm_bindgen(skip)]
    pub extensions: String,
    #[wasm_bindgen(skip)]
    pub source: String,
    #[wasm_bindgen(skip)]
    pub map: Vec<usize>,
    #[wasm_bindgen(skip)]
    pub workgroup_count: HashMap<String, [u32; 3]>,
    #[wasm_bindgen(skip)]
    pub dispatch_once: HashMap<String, bool>,
    #[wasm_bindgen(skip)]
    pub dispatch_count: HashMap<String, u32>,
    #[wasm_bindgen(skip)]
    pub assert_map: Vec<usize>,
    #[wasm_bindgen(skip)]
    pub user_data: indexmap::IndexMap<String, Vec<u32>>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self {
            extensions: String::new(),
            source: String::new(),
            map: vec![0],
            workgroup_count: HashMap::new(),
            dispatch_once: HashMap::new(),
            dispatch_count: HashMap::new(),
            assert_map: vec![],
            user_data: indexmap::IndexMap::from([("_dummy".into(), vec![0])]),
        }
    }
    fn push_line(&mut self, s: &str, n: usize) {
        self.source.push_str(s);
        self.source.push('\n');
        self.map.push(n);
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Preprocessor {
    defines: HashMap<String, String>,
    source: SourceMap,
    storage_count: usize,
    assert_count: usize,
    special_strings: bool,
}

static RE_COMMENT: Lazy<Regex> = lazy_regex!(r"(//.*|(?s:/\*.*?\*/))");
static RE_QUOTES: Lazy<Regex> = lazy_regex!(r#""((?:[^\\"]|\\.)*)""#);
static RE_CHEVRONS: Lazy<Regex> = lazy_regex!("<(.*)>");
static RE_WORD: Lazy<Regex> = lazy_regex!("[[:word:]]+");

const STRING_MAX_LEN: usize = 20;

pub fn strip_comments(s: &str) -> String {
    RE_COMMENT.replace_all(s, "").to_string()
}

impl Preprocessor {
    pub fn new(mut defines: HashMap<String, String>) -> Self {
        defines.insert("STRING_MAX_LEN".to_string(), STRING_MAX_LEN.to_string());
        Self {
            defines,
            source: SourceMap::new(),
            storage_count: 0,
            assert_count: 0,
            special_strings: false,
        }
    }

    fn subst_defines(&self, source: &str) -> String {
        RE_WORD
            .replace_all(source, |caps: &regex::Captures| {
                let name = &caps[0];
                self.defines
                    .get(name)
                    .unwrap_or(&name.to_string())
                    .to_owned()
            })
            .to_string()
    }

    async fn preprocess(&mut self, shader: &str) -> Result<(), WGSLError> {
        for (line, n) in shader.lines().zip(1..) {
            self.process_line(line, n).await?
        }
        Ok(())
    }

    #[async_recursion(?Send)]
    async fn process_line(&mut self, line_orig: &str, n: usize) -> Result<(), WGSLError> {
        let mut line = self.subst_defines(line_orig);
        if line.trim_start().starts_with("enable") {
            line = RE_COMMENT.replace(&line, "").to_string();
            self.source.extensions.push_str(&line);
            self.source.extensions.push('\n');
        } else if line.trim_start().starts_with('#') {
            line = RE_COMMENT.replace(&line, "").to_string();
            let tokens: Vec<&str> = line.trim().split(' ').collect();
            match tokens[..] {
                ["#include", name] => {
                    let include = match RE_QUOTES.captures(name) {
                        None => match RE_CHEVRONS.captures(name) {
                            None => {
                                return Err(WGSLError::new(
                                    "Path must be enclosed in quotes".to_string(),
                                    n,
                                ))
                            }
                            Some(cap) => {
                                let path = &cap[1];
                                if path == "string" {
                                    self.special_strings = true;
                                }
                                fetch_include(format!("std/{path}")).await
                            }
                        },
                        Some(cap) => fetch_include(cap[1].to_string()).await,
                    };
                    if let Some(code) = include {
                        for line in code.lines() {
                            self.process_line(line, n).await?
                        }
                    } else {
                        return Err(WGSLError::new(format!("Cannot find include {name}"), n));
                    }
                }
                ["#workgroup_count", name, x, y, z] => {
                    self.source.workgroup_count.insert(
                        name.to_string(),
                        [parse_u32(x, n)?, parse_u32(y, n)?, parse_u32(z, n)?],
                    );
                }
                ["#dispatch_once", name] => {
                    self.source.dispatch_once.insert(name.to_string(), true);
                }
                ["#dispatch_count", name, x] => {
                    self.source
                        .dispatch_count
                        .insert(name.to_string(), parse_u32(x, n)?);
                }
                ["#define", ..] => {
                    let l = line_orig
                        .trim()
                        .split(' ')
                        .nth(1)
                        .ok_or(WGSLError::new(format!("Parse error"), n))?;
                    let r = tokens[2..].join(" ");
                    if self.defines.get(l).is_some() {
                        return Err(WGSLError::new(format!("Cannot redefine {l}"), n));
                    }
                    self.defines.insert(l.to_string(), r);
                }
                ["#storage", name, ref types @ ..] => {
                    if self.storage_count >= 2 {
                        return Err(WGSLError::new(
                            "Only two storage buffers are currently supported".to_string(),
                            n,
                        ));
                    }
                    let ty = types.join(" ");
                    self.source.push_line(
                        &format!(
                            "@group(0) @binding({}) var<storage,read_write> {name}: {ty};",
                            self.storage_count
                        ),
                        n,
                    );
                    self.storage_count += 1;
                }
                ["#assert", ref counters @ ..] => {
                    if self.assert_count >= NUM_ASSERT_COUNTERS {
                        return Err(WGSLError::new(
                            format!("A maximum of {NUM_ASSERT_COUNTERS} assertions are currently supported"),
                            n,
                        ));
                    }
                    let pred = counters.join(" ");
                    self.source
                        .push_line(&format!("assert({}, {pred});", self.assert_count), n);
                    self.source.assert_map.push(n);
                    self.assert_count += 1;
                }
                ["#data", name, "u32", ref data @ ..] => {
                    match data.join("").split(',').map(|s| parse_u32(s, n)).collect() {
                        Ok::<Vec<u32>, _>(mut data) => {
                            if self.source.user_data.contains_key("_dummy") {
                                self.source.user_data.clear();
                            }
                            let name = name.to_string();
                            if let Some(arr) = self.source.user_data.get_mut(&name) {
                                arr.append(&mut data);
                            } else {
                                self.source.user_data.insert(name, data);
                            }
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                }
                _ => {
                    return Err(WGSLError::new(
                        "Unrecognised preprocessor directive".to_string(),
                        n,
                    ))
                }
            }
        } else {
            if self.special_strings {
                let mut err = None;
                line = RE_QUOTES.replace(&line, |caps: &Captures| {
                    if let Ok(s) = snailquote::unescape(&caps[0]) {
                        let mut chars: Vec<u32> = s.chars().map(|c| c as u32).collect();
                        let len = chars.len();
                        if len > STRING_MAX_LEN {
                            err = Some(WGSLError::new(
                                format!(
                                    "String literals cannot be longer than {STRING_MAX_LEN} characters"
                                ),
                                n,
                            ));
                        }
                        chars.resize(STRING_MAX_LEN, 0);
                        format!(
                            "String({len}, array<uint,{STRING_MAX_LEN}>({}))",
                            chars.iter().map(|c| format!("{c:#04x}")).join(", ")
                        )
                    } else {
                        caps[0].to_string()
                    }
                }).to_string();
                if let Some(e) = err {
                    return Err(e);
                }
            }
            self.source.push_line(&line, n);
        }
        Ok(())
    }

    pub async fn run(&mut self, shader: &str) -> Option<SourceMap> {
        match self.preprocess(shader).await {
            Ok(()) => Some(std::mem::take(&mut self.source)),
            Err(e) => {
                e.submit();
                None
            }
        }
    }
}
