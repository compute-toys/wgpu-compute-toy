use pollster::block_on;
use reqwest_middleware::ClientWithMiddleware;
use serde::{Deserialize, Serialize};
use std::error::Error;

use crate::utils::fetch;

pub struct Shader {
    pub shader: String,
    pub meta: ShaderMeta,
    pub textures: Vec<LoadedTexture>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
#[serde(rename_all = "camelCase")]
pub struct ShaderMeta {
    pub uniforms: Vec<Uniform>,
    pub textures: Vec<Texture>,
    #[serde(default)]
    pub float32_enabled: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Uniform {
    pub name: String,
    pub value: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Texture {
    pub img: String,
}

pub struct LoadedTexture {
    pub img: String,
    pub data: Vec<u8>,
}

pub fn load_shader_meta(json: String) -> Result<ShaderMeta, Box<dyn Error>> {
    Ok(serde_json::from_str(&json)?)
}

pub fn load_shader<S: Loader, T: Loader>(
    source_loader: S,
    texture_loader: T,
    name: &String,
) -> Result<Shader, String> {
    let shader_filename = format!("{name}.wgsl");
    let meta_filename = format!("{name}.wgsl.json");

    let shader_source = source_loader.load_string(&shader_filename)?;
    let meta = if let Ok(meta_json) = source_loader.load_string(&meta_filename) {
        load_shader_meta(meta_json)
            .map_err(|e| format!("error loading meta for {}: {:?}", name, e))?
    } else {
        ShaderMeta::default()
    };

    let textures = meta
        .textures
        .iter()
        .map(|t| {
            let data = texture_loader.load_bytes(&t.img)?;
            Ok(LoadedTexture {
                img: t.img.clone(),
                data,
            })
        })
        .collect::<Result<Vec<LoadedTexture>, String>>()?;

    Ok(Shader {
        shader: shader_source,
        meta,
        textures,
    })
}

pub trait Loader {
    fn load_bytes(&self, path: &String) -> Result<Vec<u8>, String>;

    fn load_string(&self, path: &String) -> Result<String, String> {
        String::from_utf8(self.load_bytes(path)?)
            .map_err(|e| format!("error reading {} as utf8: {:?}", path, e))
    }
}

pub struct FolderLoader {
    base_path: String,
}

impl FolderLoader {
    pub fn new(base_path: String) -> Self {
        Self { base_path }
    }
}

impl Loader for &FolderLoader {
    fn load_bytes(&self, path: &String) -> Result<Vec<u8>, String> {
        Ok(std::fs::read(format!("{}/{path}", self.base_path))
            .map_err(|e| format!("error including file {}: {:?}", path, e))?)
    }
}

pub struct WebLoader {
    client: ClientWithMiddleware,
}

impl WebLoader {
    pub fn new() -> Self {
        let client = reqwest_middleware::ClientBuilder::new(reqwest::Client::new())
            .with(reqwest_middleware_cache::Cache {
                mode: reqwest_middleware_cache::CacheMode::Default,
                cache_manager: reqwest_middleware_cache::managers::CACacheManager::default(),
            })
            .build();
        Self { client }
    }
}

impl Loader for &WebLoader {
    fn load_bytes(&self, path: &String) -> Result<Vec<u8>, String> {
        block_on(async {
            let url = if path.starts_with("http") {
                path.clone()
            } else {
                std::format!("https://compute.toys/{}", path)
            };
            let resp = self
                .client
                .get(&url)
                .send()
                .await
                .map_err(|e| format!("{:?}", e))?;
            Ok(resp.bytes().await.map_err(|e| format!("{:?}", e))?.to_vec())
        })
    }
}
