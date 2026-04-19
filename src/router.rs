//! Model-string → backend routing.
//!
//! Each request carries a `model` string; the router resolves it to a
//! [`Backend`] plus an upstream model name via a three-rule ladder
//! (longest-prefix wins within each rule):
//!
//! 1. [`MatchKind::ExactDefault`] — request model exactly equals a route
//!    prefix that has a `model` default → upstream receives the default.
//! 2. [`MatchKind::Namespace`] — request model has shape `prefix/tail` →
//!    upstream receives `tail`.
//! 3. [`MatchKind::RawPrefix`] — request model starts with a route prefix →
//!    upstream receives the original string unchanged.
//!
//! No fallthrough: an unmatched request returns an error to the caller.

use crate::types::Backend;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchKind {
    /// Request model string exactly equals a route's prefix *and* that route
    /// declares a default `model`. Upstream receives the route's default
    /// model instead of the literal prefix.
    ExactDefault,
    /// Request model has the form `prefix/tail`. Upstream receives `tail`.
    Namespace,
    /// Request model starts with some route's prefix (longest wins). Upstream
    /// receives the original string unchanged.
    RawPrefix,
}

impl MatchKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExactDefault => "exact-default",
            Self::Namespace => "namespace",
            Self::RawPrefix => "raw-prefix",
        }
    }
}

#[derive(Debug)]
pub struct Resolution<'a> {
    pub backend: &'a Backend,
    pub upstream_model: String,
    pub matched_by: MatchKind,
}

#[derive(Debug, Clone)]
pub struct ModelRouter {
    backends: Vec<Backend>,
    /// Indices into `backends`, sorted by `prefix.len()` desc so longer
    /// prefixes win ties during raw-prefix matching.
    prefix_order: Vec<usize>,
}

impl ModelRouter {
    pub fn new(backends: Vec<Backend>) -> Self {
        let mut prefix_order: Vec<usize> = (0..backends.len()).collect();
        prefix_order.sort_by(|&a, &b| backends[b].prefix.len().cmp(&backends[a].prefix.len()));
        Self {
            backends,
            prefix_order,
        }
    }

    pub fn resolve(&self, model: &str) -> Option<Resolution<'_>> {
        // 1) Exact + default-model swap.
        for &idx in &self.prefix_order {
            let backend = &self.backends[idx];
            if backend.prefix == model {
                if let Some(default) = backend.default_model.as_deref() {
                    return Some(Resolution {
                        backend,
                        upstream_model: default.to_string(),
                        matched_by: MatchKind::ExactDefault,
                    });
                }
            }
        }

        // 2) Namespace match — first `/` segment equals some route prefix.
        if let Some((head, tail)) = model.split_once('/') {
            for &idx in &self.prefix_order {
                let backend = &self.backends[idx];
                if backend.prefix == head {
                    return Some(Resolution {
                        backend,
                        upstream_model: tail.to_string(),
                        matched_by: MatchKind::Namespace,
                    });
                }
            }
        }

        // 3) Raw longest-prefix — forward unchanged.
        for &idx in &self.prefix_order {
            let backend = &self.backends[idx];
            if model.starts_with(&backend.prefix) {
                return Some(Resolution {
                    backend,
                    upstream_model: model.to_string(),
                    matched_by: MatchKind::RawPrefix,
                });
            }
        }

        None
    }

    /// Summarise what the router can match — used for 400 error messages.
    pub fn describe_catalog(&self) -> String {
        let entries: Vec<String> = self
            .backends
            .iter()
            .map(|b| match &b.default_model {
                Some(m) => format!("{} → {}", b.prefix, m),
                None => b.prefix.clone(),
            })
            .collect();
        format!("[{}]", entries.join(", "))
    }
}
