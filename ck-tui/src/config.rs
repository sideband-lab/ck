use anyhow::Result;
use ck_core::SearchMode;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::path::PathBuf;

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum PreviewMode {
    Heatmap, // Semantic similarity coloring
    Syntax,  // Syntax highlighting
    Chunks,  // Show chunk boundaries
}

#[derive(Serialize, Deserialize)]
pub struct TuiConfig {
    #[serde(with = "search_mode_serde")]
    pub search_mode: SearchMode,
    pub preview_mode: PreviewMode,
    pub full_file_mode: bool,
}

mod search_mode_serde {
    use super::*;

    pub fn serialize<S>(mode: &SearchMode, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = match mode {
            SearchMode::Semantic => "semantic",
            SearchMode::Regex => "regex",
            SearchMode::Hybrid => "hybrid",
            SearchMode::Lexical => "lexical",
        };
        serializer.serialize_str(s)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SearchMode, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "semantic" => SearchMode::Semantic,
            "regex" => SearchMode::Regex,
            "hybrid" => SearchMode::Hybrid,
            "lexical" => SearchMode::Lexical,
            _ => SearchMode::Semantic, // Default fallback
        })
    }
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            search_mode: SearchMode::Semantic,
            preview_mode: PreviewMode::Heatmap,
            full_file_mode: true,
        }
    }
}

impl TuiConfig {
    pub fn load() -> Self {
        let config_path = Self::config_path();
        if let Ok(contents) = std::fs::read_to_string(&config_path) {
            serde_json::from_str(&contents).unwrap_or_default()
        } else {
            Self::default()
        }
    }

    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path();
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let contents = serde_json::to_string_pretty(self)?;
        std::fs::write(&config_path, contents)?;
        Ok(())
    }

    fn config_path() -> PathBuf {
        if let Some(config_dir) = dirs::config_dir() {
            config_dir.join("ck").join("tui.json")
        } else {
            PathBuf::from(".ck_tui.json")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_search_mode_round_trips_through_config() {
        let mut config = TuiConfig::default();
        config.search_mode = SearchMode::Semantic;
        let encoded = serde_json::to_value(&config).unwrap();
        assert_eq!(encoded["search_mode"], "semantic");

        let decoded: TuiConfig = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.search_mode, SearchMode::Semantic);
        assert_eq!(decoded.preview_mode, PreviewMode::Heatmap);
        assert!(decoded.full_file_mode);
    }

    #[test]
    fn unknown_search_mode_defaults_to_semantic() {
        let config: TuiConfig = serde_json::from_str(
            r#"{"search_mode":"future-mode","preview_mode":"Syntax","full_file_mode":false}"#,
        )
        .unwrap();
        assert_eq!(config.search_mode, SearchMode::Semantic);
    }
}
