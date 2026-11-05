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
use crate::graph::direction::UndirectedMode;
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
        // its files, its algorithms' and (in tests and benchmarks, kept apart) their exact values
        #[cfg(any(test, feature = "bench"))]
        let dirs = ["", super::utils::EXACT_VALUE_CACHE_DIR];
        #[cfg(not(any(test, feature = "bench")))]
        let dirs = [""];
        for dir in dirs {
            for f in glob::glob(&format!("{CACHE_DIR}{dir}*{id}.*"))?.flatten() {
                let _ = std::fs::remove_file(f);
            }
        }
    }
    let _ = std::fs::remove_file(registry);
    Ok(())
}

/// Orders an entry's files for renaming into place: its metadata, which marks it complete, last.
/// Compared by file name: glob's paths and the cache's differ in their leading `./`.
fn metadata_last(files: &mut [PathBuf], metadata_fn: &str) {
    let metadata = Path::new(metadata_fn).file_name();
    files.sort_by_key(|f| f.file_name() == metadata);
}

/// An arc an undirected edge comes from: out of its node (by its edge id), or into it (by its
/// edge id in the transpose).
#[derive(Clone, Copy)]
enum Source {
    Out(usize),
    In(usize),
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
        // (only directed graphs with an unreciprocated arc have one)
        self.open_or_derive(TRANSPOSE, false, |transpose| {
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
            let mut t_neighbors_mem = AbstractedProceduralMemoryMut::<Ix>::from_file(
                &transpose.neighbors_file,
                edge_count,
            )?;
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

    /// The underlying undirected graph, in a given mode (see [`UndirectedMode`]): each node's
    /// out-neighbours merged with its in-neighbours (from `transpose`, this entry's transpose;
    /// self-loops counted once), keeping every arc (`Each`), one edge per neighbour (`Collapse`) or
    /// one edge per neighbour linked both ways (`Mutual`). Edges keep the label of the (first)
    /// arc they come from, out-arcs first.
    ///
    /// [`UndirectedMode`]: crate::graph::direction::UndirectedMode
    pub(crate) fn derive_undirected(
        &self,
        transpose: &Self,
        mode: UndirectedMode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        self.open_or_derive(mode.kind(), true, |undirected| {
            let node_count = self.index_bytes.saturating_sub(1);

            let out_offsets =
                AbstractedProceduralMemory::<Ix>::from_file_name(&self.offsets_filename)?;
            let out_neighbors =
                AbstractedProceduralMemory::<Ix>::from_file_name(&self.neighbors_filename)?;
            let in_offsets =
                AbstractedProceduralMemory::<Ix>::from_file_name(&transpose.offsets_filename)?;
            let in_neighbors =
                AbstractedProceduralMemory::<Ix>::from_file_name(&transpose.neighbors_filename)?;
            let (out_offsets, in_offsets) = (out_offsets.as_slice(), in_offsets.as_slice());
            let (out_neighbors, in_neighbors) = (out_neighbors.as_slice(), in_neighbors.as_slice());

            // node w's edges, in neighbour order: `emit(neighbour, arc)` for each
            let edges_of = |w: usize, emit: &mut dyn FnMut(Ix, Source)| {
                let outs = out_offsets[w].index()..out_offsets[w + 1].index();
                // in-neighbours other than w itself: its self-loops are already out-neighbours
                let ins = (in_offsets[w].index()..in_offsets[w + 1].index())
                    .filter(|&e| in_neighbors[e].index() != w);
                let (mut outs, mut ins) = (outs.peekable(), ins.peekable());
                // a run of arcs to (or from) the same neighbour: its first out-arc, first in-arc
                loop {
                    let v = match (outs.peek(), ins.peek()) {
                        (None, None) => break,
                        (Some(&o), None) => out_neighbors[o],
                        (None, Some(&i)) => in_neighbors[i],
                        (Some(&o), Some(&i)) => out_neighbors[o].min(in_neighbors[i]),
                    };
                    let (mut first_out, mut first_in) = (None, None);
                    while let Some(&o) = outs.peek().filter(|&&o| out_neighbors[o] == v) {
                        outs.next();
                        if mode == UndirectedMode::Each {
                            emit(v, Source::Out(o));
                        }
                        first_out = first_out.or(Some(o));
                    }
                    while let Some(&i) = ins.peek().filter(|&&i| in_neighbors[i] == v) {
                        ins.next();
                        if mode == UndirectedMode::Each {
                            emit(v, Source::In(i));
                        }
                        first_in = first_in.or(Some(i));
                    }
                    // (a self-loop is its own reverse)
                    let both_ways = first_out.is_some() && (first_in.is_some() || v.index() == w);
                    match mode {
                        UndirectedMode::Each => {}
                        UndirectedMode::Collapse => emit(
                            v,
                            first_out.map_or_else(|| Source::In(first_in.unwrap()), Source::Out),
                        ),
                        UndirectedMode::Mutual if both_ways => {
                            emit(v, Source::Out(first_out.unwrap()))
                        }
                        UndirectedMode::Mutual => {}
                    }
                }
            };

            let mut edge_count = 0usize;
            for w in 0..node_count {
                edges_of(w, &mut |_, _| edge_count += 1);
            }
            ensure_fits_in_index_type::<Ix>(node_count, edge_count)?;

            let mut u_offsets_mem = AbstractedProceduralMemoryMut::<Ix>::from_file(
                &undirected.offsets_file,
                node_count + 1,
            )?;
            let mut u_neighbors_mem = AbstractedProceduralMemoryMut::<Ix>::from_file(
                &undirected.neighbors_file,
                edge_count,
            )?;
            let u_offsets = u_offsets_mem.as_mut_slice();
            let u_neighbors = u_neighbors_mem.as_mut_slice();
            let mut out_labels = self.edge_label_copier(undirected, edge_count)?;
            let mut in_labels = transpose.edge_label_copier(undirected, edge_count)?;

            let mut position = 0usize;
            for w in 0..node_count {
                u_offsets[w] = Ix::new(position);
                edges_of(w, &mut |v, arc| {
                    u_neighbors[position] = v;
                    match arc {
                        Source::Out(e) => out_labels.copy(e, position),
                        Source::In(e) => in_labels.copy(e, position),
                    }
                    position += 1;
                });
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

    /// The subgraph induced by the nodes `keep` holds for, as a new entry of id `id`: its nodes
    /// numbered in their order here, with their labels and metalabels, and the arcs between them,
    /// with their labels. A failed build leaves nothing behind.
    pub(crate) fn derive_induced(
        &self,
        id: &str,
        keep: &dyn Fn(usize) -> bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let metadata = self.metadata()?;
        let dataset_name = format!("{}_induced", metadata.dataset_name);
        let mut induced = Self::init_with_id(id, self.batch)?;
        let built = (|| -> Result<(), Box<dyn std::error::Error>> {
            let node_count = self.index_bytes.saturating_sub(1);
            let offsets = AbstractedProceduralMemory::<Ix>::from_file_name(&self.offsets_filename)?;
            let neighbors =
                AbstractedProceduralMemory::<Ix>::from_file_name(&self.neighbors_filename)?;
            let (offsets, neighbors) = (offsets.as_slice(), neighbors.as_slice());

            // each kept node's new id (`usize::MAX` for the others)
            let new_id_fn =
                cache_file_name(&induced.offsets_filename, &FileType::Helper(H::H), Some(0))?;
            let mut new_id_mem = crate::shared_slice::SharedSliceMut::<usize>::abst_mem_mut(
                &new_id_fn, node_count, true,
            )?;
            let new_id = new_id_mem.as_mut_slice();
            let mut kept = 0usize;
            for (u, id) in new_id.iter_mut().enumerate() {
                *id = if keep(u) {
                    kept += 1;
                    kept - 1
                } else {
                    usize::MAX
                };
            }
            let arcs_kept = |u: usize| {
                (offsets[u].index()..offsets[u + 1].index())
                    .filter(|&e| new_id[neighbors[e].index()] != usize::MAX)
            };
            let mut edge_count = 0usize;
            for u in (0..node_count).filter(|&u| new_id[u] != usize::MAX) {
                edge_count += arcs_kept(u).count();
            }
            ensure_fits_in_index_type::<Ix>(kept, edge_count)?;

            let mut i_offsets_mem =
                AbstractedProceduralMemoryMut::<Ix>::from_file(&induced.offsets_file, kept + 1)?;
            let mut i_neighbors_mem = AbstractedProceduralMemoryMut::<Ix>::from_file(
                &induced.neighbors_file,
                edge_count,
            )?;
            let (i_offsets, i_neighbors) =
                (i_offsets_mem.as_mut_slice(), i_neighbors_mem.as_mut_slice());
            let mut labels = self.edge_label_copier(&induced, edge_count)?;
            let mut position = 0usize;
            for u in (0..node_count).filter(|&u| new_id[u] != usize::MAX) {
                i_offsets[new_id[u]] = Ix::new(position);
                for e in arcs_kept(u) {
                    // neighbours keep their (sorted) order: new ids follow old ones
                    i_neighbors[position] = Ix::new(new_id[neighbors[e].index()]);
                    labels.copy(e, position);
                    position += 1;
                }
            }
            i_offsets[kept] = Ix::new(position);
            labels.finish()?;
            i_offsets_mem.flush()?;
            i_neighbors_mem.flush()?;
            induced.index_bytes = kept + 1;
            induced.graph_bytes = edge_count;

            // node labels, of the kept nodes
            if N::is_labeled() {
                let src =
                    AbstractedProceduralMemory::<N>::from_file_name(&self.nodelabel_filename)?;
                let mut dest =
                    AbstractedProceduralMemoryMut::<N>::from_file(&induced.nodelabel_file, kept)?;
                for u in (0..node_count).filter(|&u| new_id[u] != usize::MAX) {
                    dest.as_mut_slice()[new_id[u]] = src.as_slice()[u];
                }
                dest.flush()?;
            } else {
                let mut src = OpenOptions::new()
                    .read(true)
                    .open(&self.nodelabel_filename)?;
                let mut dest = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(&induced.nodelabel_filename)?;
                std::io::copy(&mut src, &mut dest)?;
                dest.flush()?;
            }

            // metalabels, of the kept nodes, renumbered (the fst streams them in key order)
            let final_metalabel_fn = cache_file_name(
                &induced.metalabel_filename,
                &FileType::MetaLabel(H::H),
                None,
            )?;
            {
                use fst::Streamer;
                let source = fst::Map::new(unsafe {
                    memmap2::Mmap::map(&std::fs::File::open(&self.metalabel_filename)?)?
                })?;
                let mut builder = fst::MapBuilder::new(std::io::BufWriter::new(
                    crate::shared_slice::create_new_file(&final_metalabel_fn)?,
                ))?;
                let mut stream = source.stream();
                while let Some((metalabel, node)) = stream.next() {
                    let new = new_id[node as usize];
                    if new != usize::MAX {
                        builder.insert(metalabel, new as u64)?;
                    }
                }
                builder.finish()?;
            }
            induced.metalabel_filename = final_metalabel_fn;
            induced.metalabel_file = Arc::new(
                OpenOptions::new()
                    .read(true)
                    .open(&induced.metalabel_filename)?,
            );

            drop(new_id_mem);
            let _ = std::fs::remove_file(&new_id_fn);
            // a symmetric graph's induced subgraphs are (otherwise, it's checked)
            if metadata.symmetric == Some(true) {
                induced.symmetric = Some(true);
            }
            induced.finish(&dataset_name)
        })();
        if let Err(e) = built {
            let pattern = format!("{CACHE_DIR}*{}.*", id_from_filename(id)?);
            for f in glob::glob(&pattern)?.flatten() {
                let _ = std::fs::remove_file(f);
            }
            return Err(format!("error building the induced subgraph: {e}").into());
        }
        Ok(induced)
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
        symmetric: bool,
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
            derived.symmetric = Some(symmetric);
            derived.finish(&dataset_name)?;
            let mut metadata = derived.metadata()?;
            metadata.cache_id = hashed_id.clone();
            metadata.write_file(derived.metadata_filename())?;
            Ok(())
        })()
        .and_then(|()| {
            let mut files: Vec<PathBuf> = glob::glob(&format!("{CACHE_DIR}*{build_hashed_id}.*"))?
                .flatten()
                .collect();
            metadata_last(
                &mut files,
                &metadata_fn.replace(&hashed_id, &build_hashed_id),
            );
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
        let mut src = OpenOptions::new()
            .read(true)
            .open(&self.nodelabel_filename)?;
        let mut dest = OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&derived.nodelabel_filename)?;
        std::io::copy(&mut src, &mut dest)?;
        dest.flush()?;

        // the fst is built under a temporary filename and promoted to its final one once
        // finalized (see `merge_fsts`): write the final one directly
        let final_metalabel_fn = cache_file_name(
            &derived.metalabel_filename,
            &FileType::MetaLabel(H::H),
            None,
        )?;
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
            let mut src = OpenOptions::new()
                .read(true)
                .open(&self.edgelabel_filename)?;
            let mut dest = OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&derived.edgelabel_filename)?;
            std::io::copy(&mut src, &mut dest)?;
            dest.flush()?;
            return Ok(EdgeLabelCopier(None));
        }
        let src = AbstractedProceduralMemory::<E>::from_file_name(&self.edgelabel_filename)?;
        let dest =
            AbstractedProceduralMemoryMut::<E>::from_file(&derived.edgelabel_file, edge_count)?;
        Ok(EdgeLabelCopier(Some((src, dest))))
    }
}

struct EdgeLabelCopier<E>(
    Option<(
        AbstractedProceduralMemory<E>,
        AbstractedProceduralMemoryMut<E>,
    )>,
);

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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn metadata_is_renamed_last_whatever_the_path_prefix() {
        let mut files: Vec<PathBuf> = [
            "edges_h.mmap",
            "metadata_h.toml",
            "nodelabels_h.mmap",
            "index_h.mmap",
        ]
        .iter()
        .map(|f| Path::new(".test_cache").join(f))
        .collect();
        metadata_last(&mut files, "./.test_cache/metadata_h.toml");
        assert_eq!(
            files.last().unwrap(),
            Path::new(".test_cache/metadata_h.toml")
        );
        assert_eq!(files.len(), 4);
    }
}
