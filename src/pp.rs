use crate::utils::{fetch_include, parse_u32};
use lazy_regex::regex;
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
}

impl SourceMap {
    fn new() -> Self {
        Self {
            source: String::new(),
            map: vec![0],
            workgroup_count: HashMap::new(),
            dispatch_count: HashMap::new(),
        }
    }
    fn push_line(&mut self, s: &str, n: usize) {
        self.source.push_str(s);
        self.source.push('\n');
        self.map.push(n);
    }
}

pub struct Preprocessor {
    defines: HashMap<String, String>,
}

impl Preprocessor {
    pub fn new(defines: HashMap<String, String>) -> Self {
        Self { defines }
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

    async fn preprocess(&mut self, shader: &str) -> Result<SourceMap, WGSLError> {
        let mut source = SourceMap::new();
        let mut storage_count = 0;
        for (line, n) in shader.lines().zip(1..) {
            let line = self.subst_defines(line);
            if line.chars().nth(0) == Some('#') {
                let tokens: Vec<&str> = line.split(" ").collect();
                match tokens[..] {
                    ["#include", name] => {
                        let include = match regex!(r#""(.*)""#).captures(name) {
                            None => {
                                return Err(WGSLError::new(
                                    "Path must be enclosed in quotes".to_string(),
                                    n,
                                ))
                            }
                            Some(cap) => fetch_include(cap[1].to_string()).await,
                        };
                        if let Some(code) = include {
                            for line in code.lines() {
                                let line = self.subst_defines(line);
                                source.push_line(&line, n);
                            }
                        } else {
                            return Err(WGSLError::new(format!("Cannot find include {name}"), n));
                        }
                    }
                    ["#workgroup_count", name, x, y, z] => {
                        source.workgroup_count.insert(
                            name.to_string(),
                            [parse_u32(x, n)?, parse_u32(y, n)?, parse_u32(z, n)?],
                        );
                    }
                    ["#dispatch_count", name, x] => {
                        source
                            .dispatch_count
                            .insert(name.to_string(), parse_u32(x, n)?);
                    }
                    ["#define", l, r] => {
                        self.defines.insert(l.to_string(), r.to_string());
                    }
                    ["#storage", name, ty] => {
                        if storage_count >= 2 {
                            return Err(WGSLError::new(
                                "Only two storage buffers are currently supported".to_string(),
                                n,
                            ));
                        }
                        source.push_line(&format!("@group(0) @binding({storage_count}) var<storage,read_write> {name}: {ty};"), n);
                        storage_count += 1;
                    }
                    _ => {
                        return Err(WGSLError::new(
                            "Unrecognised preprocessor directive".to_string(),
                            n,
                        ))
                    }
                }
            } else {
                source.push_line(&line, n);
            }
        }
        Ok(source)
    }

    pub async fn run(&mut self, shader: &str) -> Option<SourceMap> {
        match self.preprocess(&shader).await {
            Ok(source) => Some(source),
            Err(e) => {
                e.submit();
                None
            }
        }
    }
}
