//! Graphs derived from a cached graph: its transpose and its underlying undirected graph (see
//! [`GraphMemoryMap::transpose`] and [`GraphMemoryMap::to_undirected`]). Each is a cache entry of
//! its own, whose id is derived from the original's, so that it's built once and found again.
//!
//! [`GraphMemoryMap::transpose`]: ../struct.GraphMemoryMap.html#method.transpose
//! [`GraphMemoryMap::to_undirected`]: ../struct.GraphMemoryMap.html#method.to_undirected
use super::{
    GraphCache, ensure_fits_in_index_type,
    utils::{CACHE_DIR, FileType, H, cache_file_name, id_from_filename},
};
use crate::graph::IndexType;
use crate::shared_slice::{AbstractedProceduralMemory, AbstractedProceduralMemoryMut};

use std::{
    fs::OpenOptions,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

const TRANSPOSE: &str = "transpose";
const UNDIRECTED: &str = "undirected";

/// The file listing the (hashed) ids of the graphs derived from the entry with a given (hashed)
/// id: named after it, so that it goes with the entry's files.
fn registry(cache_id: &str) -> String {
    format!("{CACHE_DIR}derived_{cache_id}.list")
}

/// Records that the entry with (hashed) id `derived` was derived from the one with `cache_id`.
fn register_derived(cache_id: &str, derived: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(registry(cache_id))?;
    writeln!(f, "{derived}")?;
    Ok(())
}

/// Removes the graphs derived from the entry with a given (hashed) id, and those derived from
/// them: an entry being dropped or (re)built makes them useless or stale.
pub(crate) fn drop_derived_of(cache_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let registry = registry(cache_id);
    let Ok(f) = std::fs::File::open(&registry) else {
        return Ok(());
    };
    let mut derived: Vec<String> = BufReader::new(f).lines().collect::<Result<_, _>>()?;
    derived.sort();
    derived.dedup();
    for id in derived.iter().filter(|id| !id.is_empty()) {
        drop_derived_of(id)?;
        for f in glob::glob(&format!("{CACHE_DIR}*{id}.*"))?.flatten() {
            let _ = std::fs::remove_file(f);
        }
    }
    let _ = std::fs::remove_file(registry);
    Ok(())
}

/// Tells apart concurrent builds of derived graphs within a process.
static BUILDS: AtomicUsize = AtomicUsize::new(0);

#[allow(dead_code)]
impl<N: super::N, E: super::E, Ix: IndexType> GraphCache<N, E, Ix> {
    /// The (unhashed) id of this entry's derived graph of a given kind, for its label and index
    /// types: instances of an entry opened with other label types derive graphs of their own.
    pub(crate) fn derived_id(&self, kind: &str) -> Result<String, Box<dyn std::error::Error>> {
        let size = |labeled: bool, size: usize| if labeled { size } else { 0 };
        Ok(format!(
            "{}_{kind}_n{}_e{}_ix{}",
            self.cache_id()?,
            size(N::is_labeled(), std::mem::size_of::<N>()),
            size(E::is_labeled(), std::mem::size_of::<E>()),
            std::mem::size_of::<Ix>()
        ))
    }

    /// The transpose: every arc reversed, keeping its label.
    pub(crate) fn derive_transpose(&self) -> Result<Self, Box<dyn std::error::Error>> {
        self.open_or_derive(TRANSPOSE, true, |transpose| {
            let node_count = self.index_bytes.saturating_sub(1);
            let edge_count = self.graph_bytes;

            let offsets = AbstractedProceduralMemory::<Ix>::from_file_name(&self.offsets_filename)?;
            let neighbors =
                AbstractedProceduralMemory::<Ix>::from_file_name(&self.neighbors_filename)?;
            let (offsets, neighbors) = (offsets.as_slice(), neighbors.as_slice());

            let mut t_offsets_mem = AbstractedProceduralMemoryMut::<Ix>::from_file(
                &transpose.offsets_file,
                node_count + 1,
            )?;
            let mut t_neighbors_mem =
                AbstractedProceduralMemoryMut::<Ix>::from_file(&transpose.neighbors_file, edge_count)?;
            let t_offsets = t_offsets_mem.as_mut_slice();
            let t_neighbors = t_neighbors_mem.as_mut_slice();

            // counting sort by destination; sources are visited in order, so each node's
            // in-neighbours come out sorted

            // in-degrees, then their prefix sums: t_offsets[v + 1] = end of v's in-neighbours
            t_offsets.fill(Ix::new(0));
            for v in &neighbors[..edge_count] {
                let slot = &mut t_offsets[v.index() + 1];
                *slot = Ix::new(slot.index() + 1);
            }
            for v in 0..node_count {
                t_offsets[v + 1] = Ix::new(t_offsets[v].index() + t_offsets[v + 1].index());
            }

            // t_offsets[v] (the start of v's in-neighbours) doubles as v's insertion cursor...
            let mut labels = self.edge_label_copier(transpose, edge_count)?;
            for u in 0..node_count {
                for e in offsets[u].index()..offsets[u + 1].index() {
                    let v = neighbors[e].index();
                    let position = t_offsets[v].index();
                    t_neighbors[position] = Ix::new(u);
                    labels.copy(e, position);
                    t_offsets[v] = Ix::new(position + 1);
                }
            }
            // ... leaving it at the end of v's in-neighbours, i.e. the start of v + 1's: shift
            for v in (1..=node_count).rev() {
                t_offsets[v] = t_offsets[v - 1];
            }
            t_offsets[0] = Ix::new(0);

            labels.finish()?;
            t_offsets_mem.flush()?;
            t_neighbors_mem.flush()?;
            transpose.index_bytes = node_count + 1;
            transpose.graph_bytes = edge_count;
            Ok(())
        })
    }

    /// The underlying undirected graph: each node's out-neighbours merged with its
    /// in-neighbours (from `transpose`, this entry's transpose), self-loops once, multiplicity
    /// kept. Edges keep the label of the arc they come from.
    pub(crate) fn derive_undirected(
        &self,
        transpose: &Self,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        self.open_or_derive(UNDIRECTED, false, |undirected| {
            let node_count = self.index_bytes.saturating_sub(1);

            let out_offsets =
                AbstractedProceduralMemory::<Ix>::from_file_name(&self.offsets_filename)?;
            let out_neighbors =
                AbstractedProceduralMemory::<Ix>::from_file_name(&self.neighbors_filename)?;
            let in_offsets =
                AbstractedProceduralMemory::<Ix>::from_file_name(&transpose.offsets_filename)?;
            let in_neighbors =
                AbstractedProceduralMemory::<Ix>::from_file_name(&transpose.neighbors_filename)?;
            let out_of = |w: usize| {
                let o = out_offsets.as_slice();
                o[w].index()..o[w + 1].index()
            };
            let in_of = |w: usize| {
                let o = in_offsets.as_slice();
                o[w].index()..o[w + 1].index()
            };
            let (out_neighbors, in_neighbors) = (out_neighbors.as_slice(), in_neighbors.as_slice());

            // a node's in-neighbours other than itself: its self-loops are already out-neighbours
            let in_others = |w: usize| in_of(w).filter(move |&e| in_neighbors[e].index() != w);
            let mut edge_count = 0usize;
            for w in 0..node_count {
                edge_count += out_of(w).len() + in_others(w).count();
            }
            ensure_fits_in_index_type::<Ix>(node_count, edge_count)?;

            let mut u_offsets_mem = AbstractedProceduralMemoryMut::<Ix>::from_file(
                &undirected.offsets_file,
                node_count + 1,
            )?;
            let mut u_neighbors_mem =
                AbstractedProceduralMemoryMut::<Ix>::from_file(&undirected.neighbors_file, edge_count)?;
            let u_offsets = u_offsets_mem.as_mut_slice();
            let u_neighbors = u_neighbors_mem.as_mut_slice();
            let mut out_labels = self.edge_label_copier(undirected, edge_count)?;
            let mut in_labels = transpose.edge_label_copier(undirected, edge_count)?;

            // merge each node's (sorted) out- and in-neighbours, out-neighbours first on ties
            let mut position = 0usize;
            for w in 0..node_count {
                u_offsets[w] = Ix::new(position);
                let mut outs = out_of(w).peekable();
                let mut ins = in_others(w).peekable();
                loop {
                    let take_out = match (outs.peek(), ins.peek()) {
                        (None, None) => break,
                        (Some(_), None) => true,
                        (None, Some(_)) => false,
                        (Some(&o), Some(&i)) => out_neighbors[o].index() <= in_neighbors[i].index(),
                    };
                    if take_out {
                        let e = outs.next().unwrap();
                        u_neighbors[position] = out_neighbors[e];
                        out_labels.copy(e, position);
                    } else {
                        let e = ins.next().unwrap();
                        u_neighbors[position] = in_neighbors[e];
                        in_labels.copy(e, position);
                    }
                    position += 1;
                }
            }
            u_offsets[node_count] = Ix::new(position);

            out_labels.finish()?;
            in_labels.finish()?;
            u_offsets_mem.flush()?;
            u_neighbors_mem.flush()?;
            undirected.index_bytes = node_count + 1;
            undirected.graph_bytes = edge_count;
            Ok(())
        })
    }

    /// Opens this entry's derived graph of a given kind if it was built before, or else builds
    /// it: `build` writes its offsets, neighbors and edge labels; node labels and the
    /// metalabel fst are this entry's (node ids are the same).
    ///
    /// It's built under a temporary id of its own and only renamed into place once complete
    /// (metadata last, which marks it complete): concurrent builders, in this process or
    /// others, never write files another one reads, and as builds are deterministic, whichever
    /// renames last leaves the same files. A failed build leaves nothing behind.
    fn open_or_derive(
        &self,
        kind: &str,
        directed: bool,
        build: impl FnOnce(&mut Self) -> Result<(), Box<dyn std::error::Error>>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let id = self.derived_id(kind)?;
        let hashed_id = id_from_filename(&id)?;
        let metadata_fn =
            Self::init_cache_file_from_id_or_random(&id, FileType::CacheMetadata(H::H), None)?;
        if Path::new(&metadata_fn).exists() {
            return Self::open(&metadata_fn, self.batch);
        }

        let dataset_name = format!("{}_{kind}", self.metadata()?.dataset_name);
        let build_id = format!(
            "{id}~{}~{}",
            std::process::id(),
            BUILDS.fetch_add(1, Ordering::Relaxed)
        );
        let build_hashed_id = id_from_filename(&build_id)?;
        let built = (|| -> Result<(), Box<dyn std::error::Error>> {
            let mut derived = Self::init_with_id(&build_id, self.batch)?;
            build(&mut derived)?;
            self.copy_node_labels_into(&mut derived)?;
            derived.directed = Some(directed);
            derived.finish(&dataset_name)?;
            let mut metadata = derived.metadata()?;
            metadata.cache_id = hashed_id.clone();
            metadata.write_file(derived.metadata_filename())?;
            Ok(())
        })()
        .and_then(|()| {
            let mut files: Vec<PathBuf> =
                glob::glob(&format!("{CACHE_DIR}*{build_hashed_id}.*"))?.flatten().collect();
            files.sort_by_key(|f| f == Path::new(&metadata_fn.replace(&hashed_id, &build_hashed_id)));
            // recorded first: if the entry is dropped meanwhile, its derived graphs still go
            register_derived(&self.cache_id()?, &hashed_id)?;
            for f in files {
                let to = f.to_string_lossy().replace(&build_hashed_id, &hashed_id);
                std::fs::rename(&f, to)?;
            }
            Ok(())
        });
        if let Err(e) = built {
            for f in glob::glob(&format!("{CACHE_DIR}*{build_hashed_id}.*"))?.flatten() {
                let _ = std::fs::remove_file(f);
            }
            return Err(format!("error building the graph's {kind}: {e}").into());
        }
        #[cfg(test)]
        crate::graph::test_utils::register_cache_id(hashed_id);
        Self::open(&metadata_fn, self.batch)
    }

    /// Copies node labels and the metalabel fst, which only depend on node ids.
    fn copy_node_labels_into(&self, derived: &mut Self) -> Result<(), Box<dyn std::error::Error>> {
        let mut src = OpenOptions::new().read(true).open(&self.nodelabel_filename)?;
        let mut dest = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&derived.nodelabel_filename)?;
        std::io::copy(&mut src, &mut dest)?;
        dest.flush()?;

        // the fst is built under a temporary filename and promoted to its final one once
        // finalized (see `merge_fsts`): write the final one directly
        let final_metalabel_fn =
            cache_file_name(&derived.metalabel_filename, &FileType::MetaLabel(H::H), None)?;
        std::fs::copy(&self.metalabel_filename, &final_metalabel_fn)?;
        derived.metalabel_filename = final_metalabel_fn;
        derived.metalabel_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .open(&derived.metalabel_filename)?,
        );
        Ok(())
    }

    /// Copies edge labels from this entry into `derived`'s, edge by edge (a no-op for unlabeled
    /// graphs, whose label file is copied as is).
    fn edge_label_copier(
        &self,
        derived: &Self,
        edge_count: usize,
    ) -> Result<EdgeLabelCopier<E>, Box<dyn std::error::Error>> {
        if !E::is_labeled() {
            let mut src = OpenOptions::new().read(true).open(&self.edgelabel_filename)?;
            let mut dest = OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&derived.edgelabel_filename)?;
            std::io::copy(&mut src, &mut dest)?;
            dest.flush()?;
            return Ok(EdgeLabelCopier(None));
        }
        let src = AbstractedProceduralMemory::<E>::from_file_name(&self.edgelabel_filename)?;
        let dest = AbstractedProceduralMemoryMut::<E>::from_file(&derived.edgelabel_file, edge_count)?;
        Ok(EdgeLabelCopier(Some((src, dest))))
    }
}

struct EdgeLabelCopier<E>(Option<(AbstractedProceduralMemory<E>, AbstractedProceduralMemoryMut<E>)>);

impl<E: Copy> EdgeLabelCopier<E> {
    #[inline(always)]
    fn copy(&mut self, from: usize, to: usize) {
        if let Some((src, dest)) = &mut self.0 {
            dest.as_mut_slice()[to] = src.as_slice()[from];
        }
    }

    fn finish(self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some((_, dest)) = self.0 {
            dest.flush()?;
        }
        Ok(())
    }
}
