use crate::{
    bind::NUM_ASSERT_COUNTERS,
    utils::{fetch_include, parse_u32},
};
use async_recursion::async_recursion;
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
        wgsl_error_handler(summary, row, col)
    }
    pub fn submit(&self) {
        Self::handler(&self.summary, self.line, 0)
    }
}

#[wasm_bindgen]
pub struct SourceMap {
    #[wasm_bindgen(skip)]
    pub source: String,
    #[wasm_bindgen(skip)]
    pub map: Vec<usize>,
    #[wasm_bindgen(skip)]
    pub workgroup_count: HashMap<String, [u32; 3]>,
    #[wasm_bindgen(skip)]
    pub dispatch_count: HashMap<String, u32>,
    #[wasm_bindgen(skip)]
    pub assert_map: Vec<usize>,
}

impl SourceMap {
    pub fn new() -> Self {
        Self {
            source: String::new(),
            map: vec![0],
            workgroup_count: HashMap::new(),
            dispatch_count: HashMap::new(),
            assert_map: vec![],
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
}

static RE_COMMENT: Lazy<Regex> = lazy_regex!("//.*");
static RE_QUOTES: Lazy<Regex> = lazy_regex!(r#""(.*)""#);
static RE_CHEVRONS: Lazy<Regex> = lazy_regex!("<(.*)>");

impl Preprocessor {
    pub fn new(defines: HashMap<String, String>) -> Self {
        Self {
            defines,
            source: SourceMap::new(),
            storage_count: 0,
            assert_count: 0,
        }
    }

    fn subst_defines(&self, source: &str) -> String {
        regex!("[[:word:]]+")
            .replace_all(source, |caps: &regex::Captures| match &caps[0] {
                name => self
                    .defines
                    .get(name)
                    .unwrap_or(&name.to_string())
                    .to_owned(),
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
    async fn process_line(&mut self, line: &str, n: usize) -> Result<(), WGSLError> {
        let line = self.subst_defines(line);
        if line.trim().chars().nth(0) == Some('#') {
            let line = RE_COMMENT.replace(&line, "");
            let tokens: Vec<&str> = line.trim().split(" ").collect();
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
                                fetch_include(format!("std/{path}")).await
                            }
                        },
                        Some(cap) => fetch_include(cap[1].to_string()).await,
                    };
                    if let Some(code) = include {
                        for line in code.lines() {
                            let line = self.subst_defines(line);
                            self.process_line(&line, n).await?
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
                ["#dispatch_count", name, x] => {
                    self.source
                        .dispatch_count
                        .insert(name.to_string(), parse_u32(x, n)?);
                }
                ["#define", l, r] => {
                    self.defines.insert(l.to_string(), r.to_string());
                }
                ["#storage", name, ty] => {
                    if self.storage_count >= 2 {
                        return Err(WGSLError::new(
                            "Only two storage buffers are currently supported".to_string(),
                            n,
                        ));
                    }
                    self.source.push_line(
                        &format!(
                            "@group(0) @binding({}) var<storage,read_write> {name}: {ty};",
                            self.storage_count
                        ),
                        n,
                    );
                    self.storage_count += 1;
                }
                ["#assert", ..] => {
                    if self.assert_count >= NUM_ASSERT_COUNTERS {
                        return Err(WGSLError::new(
                            format!("A maximum of {NUM_ASSERT_COUNTERS} assertions are currently supported"),
                            n,
                        ));
                    }
                    let pred = tokens[1..].join(" ");
                    self.source
                        .push_line(&format!("assert({}, {pred});", self.assert_count), n);
                    self.source.assert_map.push(n);
                    self.assert_count += 1;
                }
                _ => {
                    return Err(WGSLError::new(
                        "Unrecognised preprocessor directive".to_string(),
                        n,
                    ))
                }
            }
        } else {
            self.source.push_line(&line, n);
        }
        Ok(())
    }

    pub async fn run(&mut self, shader: &str) -> Option<SourceMap> {
        match self.preprocess(&shader).await {
            Ok(()) => Some(std::mem::take(&mut self.source)),
            Err(e) => {
                e.submit();
                None
            }
        }
    }
}
