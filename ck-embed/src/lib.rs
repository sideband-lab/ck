use anyhow::{Result, bail};
use ck_models::{ModelConfig, ModelRegistry};
#[cfg(feature = "fastembed")]
use std::path::Path;
#[cfg(any(feature = "fastembed", feature = "mixedbread"))]
use std::path::PathBuf;

pub mod reranker;
pub mod tokenizer;

pub use reranker::{
    RerankResult, Reranker, create_reranker, create_reranker_for_config,
    create_reranker_with_progress,
};
pub use tokenizer::TokenEstimator;

#[cfg(feature = "mixedbread")]
mod mixedbread;
#[cfg(feature = "mixedbread")]
use mixedbread::MixedbreadEmbedder;

pub trait Embedder: Send + Sync {
    fn id(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn model_name(&self) -> &str;
    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub type ModelDownloadCallback = Box<dyn Fn(&str) + Send + Sync>;

#[cfg(any(feature = "fastembed", feature = "mixedbread"))]
pub(crate) fn model_cache_root() -> Result<PathBuf> {
    let base = if let Some(cache_home) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(cache_home).join("ck")
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache").join("ck")
    } else if let Some(appdata) = std::env::var_os("LOCALAPPDATA") {
        PathBuf::from(appdata).join("ck").join("cache")
    } else {
        PathBuf::from(".ck_models")
    };

    Ok(base.join("models"))
}

pub fn create_embedder(model_name: Option<&str>) -> Result<Box<dyn Embedder>> {
    create_embedder_with_progress(model_name, None)
}

pub fn create_embedder_with_progress(
    model_name: Option<&str>,
    progress_callback: Option<ModelDownloadCallback>,
) -> Result<Box<dyn Embedder>> {
    let registry = ModelRegistry::default();
    let (_, config) = registry.resolve(model_name)?;
    create_embedder_for_config(&config, progress_callback)
}

#[allow(clippy::needless_return)]
pub fn create_embedder_for_config(
    config: &ModelConfig,
    progress_callback: Option<ModelDownloadCallback>,
) -> Result<Box<dyn Embedder>> {
    match config.provider.as_str() {
        "fastembed" => {
            #[cfg(feature = "fastembed")]
            {
                return Ok(Box::new(FastEmbedder::new_with_progress(
                    config.name.as_str(),
                    progress_callback,
                )?));
            }

            #[cfg(not(feature = "fastembed"))]
            {
                if let Some(callback) = progress_callback.as_ref() {
                    callback("fastembed provider unavailable; using dummy embedder");
                }
                return Ok(Box::new(DummyEmbedder::new_with_model(
                    config.name.as_str(),
                )));
            }
        }
        "mixedbread" => {
            #[cfg(feature = "mixedbread")]
            {
                return Ok(Box::new(MixedbreadEmbedder::new(
                    config,
                    progress_callback,
                )?));
            }
            #[cfg(not(feature = "mixedbread"))]
            {
                bail!(
                    "Model '{}' requires the `mixedbread` feature. Rebuild ck with Mixedbread support.",
                    config.name
                );
            }
        }
        provider => bail!("Unsupported embedding provider '{provider}'"),
    }
}

pub struct DummyEmbedder {
    dim: usize,
    model_name: String,
}

impl Default for DummyEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl DummyEmbedder {
    pub fn new() -> Self {
        Self {
            dim: 384, // Match default BGE model
            model_name: "dummy".to_string(),
        }
    }

    pub fn new_with_model(model_name: &str) -> Self {
        Self {
            dim: 384, // Match default BGE model
            model_name: model_name.to_string(),
        }
    }
}

impl Embedder for DummyEmbedder {
    fn id(&self) -> &'static str {
        "dummy"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; self.dim]).collect())
    }
}

#[cfg(feature = "fastembed")]
pub struct FastEmbedder {
    model: fastembed::TextEmbedding,
    dim: usize,
    model_name: String,
}

#[cfg(feature = "fastembed")]
impl FastEmbedder {
    pub fn new(model_name: &str) -> Result<Self> {
        Self::new_with_progress(model_name, None)
    }

    pub fn new_with_progress(
        model_name: &str,
        progress_callback: Option<ModelDownloadCallback>,
    ) -> Result<Self> {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

        let model = match model_name {
            // Current models
            "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
            "sentence-transformers/all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,

            // Enhanced models with longer context
            "nomic-embed-text-v1" => EmbeddingModel::NomicEmbedTextV1,
            "nomic-embed-text-v1.5" => EmbeddingModel::NomicEmbedTextV15,
            "jina-embeddings-v2-base-code" => EmbeddingModel::JinaEmbeddingsV2BaseCode,

            // BGE variants
            "BAAI/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
            "BAAI/bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,

            // Default to Nomic v1.5 for better performance
            _ => EmbeddingModel::NomicEmbedTextV15,
        };

        // Configure permanent model cache directory
        let model_cache_dir = model_cache_root()?;
        std::fs::create_dir_all(&model_cache_dir)?;

        if let Some(ref callback) = progress_callback {
            callback(&format!("Initializing model: {model_name}"));

            // Check if model already exists
            let model_exists = Self::check_model_exists(&model_cache_dir, model_name);
            if !model_exists {
                callback(&format!(
                    "Downloading model {} to {}",
                    model_name,
                    model_cache_dir.display()
                ));
            } else {
                callback(&format!("Using cached model: {model_name}"));
            }
        }

        // Configure max_length based on model capacity
        let max_length = match model {
            // Small models - keep at 512
            EmbeddingModel::BGESmallENV15 | EmbeddingModel::AllMiniLML6V2 => 512,
            EmbeddingModel::BGEBaseENV15 => 512,

            // Large context models - use their full capacity!
            EmbeddingModel::NomicEmbedTextV1 | EmbeddingModel::NomicEmbedTextV15 => 8192,
            EmbeddingModel::JinaEmbeddingsV2BaseCode => 8192,

            // BGE large can handle more
            EmbeddingModel::BGELargeENV15 => 512, // Conservative for BGE

            _ => 512, // Safe default
        };

        let init_options = InitOptions::new(model.clone())
            .with_show_download_progress(progress_callback.is_some())
            .with_cache_dir(model_cache_dir)
            .with_max_length(max_length);

        let embedding = TextEmbedding::try_new(init_options)?;

        if let Some(ref callback) = progress_callback {
            callback("Model loaded successfully");
        }

        let dim = match model {
            // Small models (384 dimensions)
            EmbeddingModel::BGESmallENV15 => 384,
            EmbeddingModel::AllMiniLML6V2 => 384,

            // Large context models (768 dimensions)
            EmbeddingModel::NomicEmbedTextV1 => 768,
            EmbeddingModel::NomicEmbedTextV15 => 768,
            EmbeddingModel::JinaEmbeddingsV2BaseCode => 768,
            EmbeddingModel::BGEBaseENV15 => 768,

            // Large models (1024 dimensions)
            EmbeddingModel::BGELargeENV15 => 1024,

            _ => 384, // Default to 384 for BGE default
        };

        Ok(Self {
            model: embedding,
            dim,
            model_name: model_name.to_string(),
        })
    }

    fn check_model_exists(cache_dir: &Path, model_name: &str) -> bool {
        // Simple heuristic - check if model directory exists
        let model_dir = cache_dir.join(model_name.replace("/", "_"));
        model_dir.exists()
    }
}

#[cfg(feature = "fastembed")]
impl Embedder for FastEmbedder {
    fn id(&self) -> &'static str {
        "fastembed"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();
        let embeddings = self.model.embed(text_refs, None)?;
        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_embedder() {
        let mut embedder = DummyEmbedder::new();

        assert_eq!(embedder.id(), "dummy");
        assert_eq!(embedder.dim(), 384);

        let texts = vec!["hello".to_string(), "world".to_string()];
        let embeddings = embedder.embed(&texts).unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);
        assert_eq!(embeddings[1].len(), 384);

        // Dummy embedder should return all zeros
        assert!(embeddings[0].iter().all(|&x| x == 0.0));
        assert!(embeddings[1].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_create_embedder_dummy() {
        #[cfg(not(feature = "fastembed"))]
        {
            let embedder = create_embedder(None).unwrap();
            assert_eq!(embedder.id(), "dummy");
            assert_eq!(embedder.dim(), 384);
        }
    }

    #[test]
    fn test_embedder_trait_object() {
        let mut embedder: Box<dyn Embedder> = Box::new(DummyEmbedder::new());

        let texts = vec!["test".to_string()];
        let result = embedder.embed(&texts);
        assert!(result.is_ok());

        let embeddings = result.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 384);
    }

    #[test]
    fn test_create_embedder_rejects_unknown_alias() {
        let error = create_embedder(Some("not-a-model-alias")).err().unwrap();
        assert!(
            error
                .to_string()
                .contains("Unknown model 'not-a-model-alias'")
        );
    }

    #[test]
    fn test_create_embedder_rejects_unknown_provider() {
        let config = ModelConfig {
            name: "test-model".to_string(),
            provider: "unknown-provider".to_string(),
            dimensions: 8,
            max_tokens: 16,
            description: "test configuration".to_string(),
        };

        let error = create_embedder_for_config(&config, None).err().unwrap();
        assert!(error.to_string().contains("Unsupported embedding provider"));
    }

    #[cfg(all(feature = "fastembed", feature = "mixedbread"))]
    #[test]
    fn every_registered_alias_constructs_and_embeds_ragged_inputs() {
        use std::sync::{Arc, Mutex};

        let registry = ModelRegistry::default();
        let texts = [
            "Rust errors use Result and Option for explicit recovery.".to_string(),
            "A very short query.".to_string(),
        ];

        for alias in registry.aliases() {
            let config = registry.get_model(&alias).unwrap();
            let progress = Arc::new(Mutex::new(Vec::new()));
            let progress_copy = Arc::clone(&progress);
            let mut embedder = create_embedder_with_progress(
                Some(&alias),
                Some(Box::new(move |message| {
                    progress_copy.lock().unwrap().push(message.to_string());
                })),
            )
            .unwrap_or_else(|error| panic!("failed to initialize '{alias}': {error:#}"));

            assert_eq!(embedder.model_name(), config.name, "alias {alias}");
            assert_eq!(embedder.dim(), config.dimensions, "alias {alias}");
            assert!(!progress.lock().unwrap().is_empty(), "alias {alias}");

            let embeddings = embedder
                .embed(&texts)
                .unwrap_or_else(|error| panic!("'{alias}' inference failed: {error:#}"));
            assert_eq!(embeddings.len(), texts.len(), "alias {alias}");
            for embedding in embeddings {
                assert_eq!(embedding.len(), config.dimensions, "alias {alias}");
                assert!(
                    embedding.iter().all(|value| value.is_finite()),
                    "alias {alias}"
                );
                assert!(embedding.iter().any(|value| *value != 0.0), "alias {alias}");
            }
        }
    }

    #[cfg(not(feature = "fastembed"))]
    #[test]
    fn fastembed_provider_fallback_reports_progress_and_uses_dummy_embedder() {
        use std::sync::{Arc, Mutex};

        let messages = Arc::new(Mutex::new(Vec::new()));
        let messages_copy = Arc::clone(&messages);
        let embedder = create_embedder_with_progress(
            Some("bge-small"),
            Some(Box::new(move |message| {
                messages_copy.lock().unwrap().push(message.to_string());
            })),
        )
        .unwrap();

        assert_eq!(embedder.id(), "dummy");
        assert_eq!(embedder.model_name(), "BAAI/bge-small-en-v1.5");
        assert!(
            messages
                .lock()
                .unwrap()
                .iter()
                .any(|message| message.contains("using dummy embedder"))
        );
    }

    #[test]
    fn test_embedder_empty_input() {
        let mut embedder = DummyEmbedder::new();
        let texts: Vec<String> = vec![];
        let embeddings = embedder.embed(&texts).unwrap();
        assert_eq!(embeddings.len(), 0);
    }

    #[test]
    fn test_embedder_single_text() {
        let mut embedder = DummyEmbedder::new();
        let texts = vec!["single text".to_string()];
        let embeddings = embedder.embed(&texts).unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 384);
    }

    #[test]
    fn test_embedder_multiple_texts() {
        let mut embedder = DummyEmbedder::new();
        let texts = vec![
            "first text".to_string(),
            "second text".to_string(),
            "third text".to_string(),
        ];
        let embeddings = embedder.embed(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for embedding in &embeddings {
            assert_eq!(embedding.len(), 384);
        }
    }
}
