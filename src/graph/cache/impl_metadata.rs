use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use super::{CacheMetadata, ensure_file_writable};
use crate::utils::type_of;

#[allow(dead_code)]
impl CacheMetadata {
    pub const VERSION: u16 = 2;
    pub const GRAPH_CACHE_CACHE_METADATA_TOML_HEADER: &str = "[GraphCache.CacheMetadata]";

    #[allow(clippy::too_many_arguments)]
    pub fn now<N, E, Ix>(
        dataset_name: &str,
        cache_id: &str,
        nodes: usize,
        edges: usize,
        index_size: usize,
        node_labeled: bool,
        edge_labeled: bool,
        has_fst: bool,
        node_label_size: usize,
        edge_label_size: usize,
        tool_version: Option<String>,
        directed: Option<bool>,
    ) -> Self {
        let created_unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            format_version: Self::VERSION,
            dataset_name: dataset_name.to_string(),
            cache_id: cache_id.to_string(),
            nodes,
            edges,
            index_type: type_of::<Ix>().to_string(),
            index_size,
            node_labeled,
            edge_labeled,
            has_fst,
            node_label_type: type_of::<N>().to_string(),
            node_label_size,
            edge_label_type: type_of::<E>().to_string(),
            edge_label_size,
            created_unix_secs,
            tool_version,
            directed,
        }
    }

    // --- very small “key: value” text format ---------------------------------
    pub fn to_toml(&self) -> String {
        // Strings are written verbatim after the first " = " on the line.
        // No escaping needed on Unix paths; lines never contain newlines.
        let mut s = Self::GRAPH_CACHE_CACHE_METADATA_TOML_HEADER.to_string() + "\n";
        macro_rules! line {
            ($k:expr, $v:expr) => {{
                s.push_str($k);
                s.push_str(" = ");
                s.push_str($v);
                s.push('\n');
            }};
        }
        line!("format_version", &self.format_version.to_string());
        line!("dataset_name", &self.dataset_name);
        line!("cache_id", &self.cache_id);
        line!("nodes", &self.nodes.to_string());
        line!("edges", &self.edges.to_string());
        line!("index_type", &self.index_type);
        line!("index_size", &self.index_size.to_string());
        line!(
            "node_labeled",
            if self.node_labeled { "true" } else { "false" }
        );
        line!(
            "edge_labeled",
            if self.edge_labeled { "true" } else { "false" }
        );
        line!("has_fst", if self.has_fst { "true" } else { "false" });
        line!("node_label_type", &self.node_label_type);
        line!("node_label_size", &self.node_label_size.to_string());
        line!("edge_label_type", &self.edge_label_type);
        line!("edge_label_size", &self.edge_label_size.to_string());
        line!("created_unix_secs", &self.created_unix_secs.to_string());
        line!("tool_version", self.tool_version.as_deref().unwrap_or(""));
        if let Some(directed) = self.directed {
            line!("directed", if directed { "true" } else { "false" });
        }
        s
    }

    pub fn from_toml(text: &str) -> Result<Self, Box<dyn std::error::Error>> {
        use std::collections::HashMap;
        let mut kv = HashMap::<&str, &str>::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty()
                || line.starts_with('#')
                || line == Self::GRAPH_CACHE_CACHE_METADATA_TOML_HEADER
            {
                continue;
            }
            let (k, v) = line.split_once('=').ok_or("bad metadata line (no `:`)")?;
            kv.insert(k.trim(), v.trim());
        }
        let req = |k| -> Result<&str, Box<dyn std::error::Error>> {
            kv.get(k)
                .copied()
                .ok_or_else(|| format!("missing key `{k}`").into())
        };
        let parse_bool = |s: &str| -> Result<bool, Box<dyn std::error::Error>> {
            match s {
                "true" => Ok(true),
                "false" => Ok(false),
                _ => Err(format!("bad bool `{s}`").into()),
            }
        };

        Ok(Self {
            format_version: req("format_version")?.parse()?,
            dataset_name: req("dataset_name")?.to_string(),
            cache_id: req("cache_id")?.to_string(),
            nodes: req("nodes")?.parse()?,
            edges: req("edges")?.parse()?,
            index_type: req("index_type")?.to_string(),
            index_size: req("index_size")?.parse()?,
            node_labeled: parse_bool(req("node_labeled")?)?,
            edge_labeled: parse_bool(req("edge_labeled")?)?,
            has_fst: parse_bool(req("has_fst")?)?,
            node_label_type: req("node_label_type")?.parse()?,
            node_label_size: req("node_label_size")?.parse()?,
            edge_label_type: req("edge_label_type")?.parse()?,
            edge_label_size: req("edge_label_size")?.parse()?,
            created_unix_secs: req("created_unix_secs")?.parse()?,
            tool_version: Some(kv.get("tool_version").copied().unwrap_or("").to_string())
                .filter(|s| !s.is_empty()),
            // absent from entries built before format version 2
            directed: kv.get("directed").map(|d| parse_bool(d)).transpose()?,
        })
    }

    pub fn write_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        ensure_file_writable(path.as_ref())?;
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        f.write_all(self.to_toml().as_bytes())?;
        f.flush()?;
        Ok(())
    }

    pub fn read_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut s = String::new();
        OpenOptions::new()
            .read(true)
            .open(path)?
            .read_to_string(&mut s)?;
        Self::from_toml(&s)
    }
}

impl std::fmt::Display for CacheMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn yn(b: bool) -> &'static str {
            if b { "yes" } else { "no" }
        }
        let tool_ver = self.tool_version.as_deref().unwrap_or("—");

        writeln!(f, "CacheMetadata {{")?;
        writeln!(f, "  format_version     : {}", self.format_version)?;
        writeln!(f, "  dataset_name       : {}", self.dataset_name)?;
        writeln!(f, "  cache_id           : {}", self.cache_id)?;
        writeln!(f, "  nodes              : {}", self.nodes)?;
        writeln!(f, "  edges              : {}", self.edges)?;
        writeln!(
            f,
            "  index_type         : {} ({} bytes)",
            self.index_type, self.index_size
        )?;
        writeln!(f, "  node_labeled       : {}", yn(self.node_labeled))?;
        writeln!(f, "  edge_labeled       : {}", yn(self.edge_labeled))?;
        writeln!(f, "  has_fst            : {}", yn(self.has_fst))?;
        writeln!(
            f,
            "  node_label_type    : {} ({} bytes)",
            self.node_label_type, self.node_label_size
        )?;
        writeln!(
            f,
            "  edge_label_type    : {} ({} bytes)",
            self.edge_label_type, self.edge_label_size
        )?;
        writeln!(f, "  created_unix_secs  : {}", self.created_unix_secs)?;
        writeln!(f, "  tool_version       : {}", tool_ver)?;
        let directed = match self.directed {
            Some(true) => "yes",
            Some(false) => "no",
            None => "unknown",
        };
        writeln!(f, "  directed           : {}", directed)?;
        write!(f, "}}")
    }
}
