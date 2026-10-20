use std::{
    fs::File,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::Path,
    sync::Arc,
};

use crossbeam::thread;
use portable_atomic::{AtomicUsize, Ordering};

use super::{GraphCache, MtxField, MtxHeader, MtxSymmetry};

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> GraphCache<N, E, Ix> {
    /// Whether a MatrixMarket file's header declares it `symmetric`, `skew-symmetric` or
    /// `hermitian`: every entry is then stored in both directions, so the graph is symmetric by
    /// construction.
    pub(super) fn mtx_header_is_symmetric<P: AsRef<Path>>(
        path: P,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        Ok(Self::parse_mtx_header(path)?.symmetry.mirrored())
    }

    /// Parses and checks a MatrixMarket file's header: its banner (`%%MatrixMarket matrix
    /// coordinate <field> <symmetry>`, case insensitive), comments, and size line.
    pub(super) fn parse_mtx_header<P: AsRef<Path>>(
        path: P,
    ) -> Result<MtxHeader, Box<dyn std::error::Error>> {
        let mut rdr = BufReader::new(File::open(path.as_ref())?);

        let mut banner = String::new();
        rdr.read_line(&mut banner)?;
        let toks: Vec<String> = banner
            .split_whitespace()
            .map(str::to_ascii_lowercase)
            .collect();
        if toks.first().map(String::as_str) != Some("%%matrixmarket") {
            return Err(
                "error invalid MatrixMarket file: it must start with `%%MatrixMarket`".into(),
            );
        }
        if toks.len() < 5 || toks[1] != "matrix" {
            return Err(format!("error malformed MatrixMarket banner {:?}", banner.trim()).into());
        }
        if toks[2] != "coordinate" {
            return Err(format!(
                "error only sparse (`coordinate`) MatrixMarket matrices are graphs, not `{}` ones",
                toks[2]
            )
            .into());
        }
        let field = match toks[3].as_str() {
            "pattern" => MtxField::Pattern,
            "integer" => MtxField::Integer,
            "real" | "double" => MtxField::Real,
            "complex" => MtxField::Complex,
            other => {
                return Err(format!(
                    "error unknown MatrixMarket field `{other}`: expected `pattern`, `integer`, `real`, `double` or `complex`"
                )
                .into());
            }
        };
        let symmetry = match toks[4].as_str() {
            "general" => MtxSymmetry::General,
            "symmetric" => MtxSymmetry::Symmetric,
            "skew-symmetric" => MtxSymmetry::SkewSymmetric,
            "hermitian" => MtxSymmetry::Hermitian,
            other => {
                return Err(format!(
                    "error unknown MatrixMarket symmetry `{other}`: expected `general`, `symmetric`, `skew-symmetric` or `hermitian`"
                )
                .into());
            }
        };

        // comments (`%`) and blank lines up to the size line, `rows columns entries`
        let mut line = String::new();
        let (nrows, ncols, nnz) = loop {
            line.clear();
            if rdr.read_line(&mut line)? == 0 {
                return Err("error MatrixMarket file ends before its size line".into());
            }
            let t = line.trim();
            if t.is_empty() || t.starts_with('%') {
                continue;
            }
            let mut it = t.split_whitespace();
            let mut size = |what: &str| -> Result<usize, Box<dyn std::error::Error>> {
                let tok = it
                    .next()
                    .ok_or_else(|| format!("error MatrixMarket size line {t:?} lacks {what}"))?;
                tok.parse()
                    .map_err(|_| format!("error MatrixMarket {what} {tok:?} isn't a number").into())
            };
            break (size("rows")?, size("columns")?, size("entries")?);
        };
        if nrows != ncols {
            return Err(format!(
                "error a graph's MatrixMarket matrix must be square (nodes by nodes), but this one is {nrows} by {ncols}"
            )
            .into());
        }
        Ok(MtxHeader {
            field,
            symmetry,
            nodes: nrows,
            entries: nnz,
            data_offset: rdr.stream_position()?,
        })
    }

    pub(super) fn parallel_no_labels_parse_mtx_with<F, P: AsRef<Path>>(
        path: P,
        emit: F,
        threads: usize,
    ) -> Result<usize, Box<dyn std::error::Error>>
    where
        F: FnMut(usize, usize) + Send + Sync + Clone,
    {
        let header = Self::parse_mtx_header(path.as_ref())?;
        let file_len = std::fs::metadata(path.as_ref())?.len();
        let (nrows, ncols, nnz_declared) = (header.nodes, header.nodes, header.entries);
        let header_offset = header.data_offset;
        // mirrored entries: each stands for an arc both ways
        let symmetric = header.symmetry.mirrored();
        let thread_load = file_len
            .saturating_sub(header_offset)
            .div_ceil(threads as u64);

        let seen = Arc::new(AtomicUsize::new(0));
        let edges_emitted = Arc::new(AtomicUsize::new(0));
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::new();

            for tid in 0..threads {
                let mut local_seen = 0;
                let mut local_edges_emitted = 0usize;
                let begin_pos = std::cmp::min(header_offset + tid as u64 * thread_load, file_len);
                let end_pos =
                    std::cmp::min(header_offset + (tid + 1) as u64 * thread_load, file_len)
                        as usize;
                let p = path.as_ref();

                let seen = seen.clone();
                let edges_emitted = edges_emitted.clone();
                let mut emit = emit.clone();

                let handle = s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        // open the file separately for each thread
                        let file = File::open(p)?;
                        let mut cur_pos = begin_pos as usize;

                        // line up bufreader to next line start
                        let mut rdr = BufReader::new(file);
                        if tid == 0 {
                            rdr.seek(SeekFrom::Start(begin_pos))?;
                        } else {
                            // this way we avoid the case where thread's bufreader startsoff
                            // aligned and the first edge is skipped
                            rdr.seek(SeekFrom::Start(begin_pos.saturating_sub(1)))?;
                            cur_pos += rdr.skip_until(b'\n')?.saturating_sub(1);
                        }

                        let mut line = String::new();

                        // the whole byte range, whatever the header declares: both passes over the
                        // file must read the same entries (the total is checked against it after)
                        while end_pos > cur_pos {
                            line.clear();

                            let n = rdr.read_line(&mut line)?;
                            if n == 0 {
                                break;
                            } else {
                                cur_pos += n;
                            }

                            let t = line.trim();
                            if t.is_empty() || t.starts_with('%') {
                                continue;
                            }

                            let mut it = t.split_whitespace();

                            // 1st two tokens must be i j (1-based)
                            let i1: usize = match it.next() {
                                Some(s) => s.parse()?,
                                None => continue, // malformed line; skip
                            };

                            let j1: usize = match it.next() {
                                Some(s) => s.parse()?,
                                None => continue, // malformed line; skip
                            };

                            // Convert to 0-based, also ensure in-bounds
                            if i1 == 0 || j1 == 0 {
                                return Err(format!(
                                    "MatrixMarket indices are 1-based; found 0 ({line})"
                                )
                                .into());
                            }

                            let i = i1 - 1;
                            let j = j1 - 1;

                            if i >= nrows || j >= ncols {
                                return Err(format!(
                                    "Entry out of bounds: ({i},{j}) not in [{nrows}x{ncols}]"
                                )
                                .into());
                            }

                            // Emit edges:
                            // first directed edge i --(w)--> j
                            emit(i, j);
                            local_edges_emitted += 1;

                            // second directed edge (if symmetrical) j --(w)--> i
                            if symmetric && i != j {
                                emit(j, i);
                                local_edges_emitted += 1;
                            }

                            local_seen += 1;
                        }

                        seen.add(local_seen, Ordering::Relaxed);
                        edges_emitted.add(local_edges_emitted, Ordering::Relaxed);
                        Ok(())
                    },
                );

                handles.push(handle);
            }

            for (idx, handle) in handles.into_iter().enumerate() {
                handle
                    .join()
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error joining thread {idx}: {:?}", e).into()
                    })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in thread {idx}: {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        let entries = seen.load(Ordering::Relaxed);
        if entries != nnz_declared {
            return Err(format!(
                "error the MatrixMarket file declares {nnz_declared} entries, but has {entries}"
            )
            .into());
        }

        Ok(edges_emitted.load(Ordering::Relaxed))
    }

    pub(super) fn parallel_edge_labels_parse_mtx_with<F, P: AsRef<Path>>(
        path: P,
        emit: F,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, usize, usize) + Send + Sync + Clone,
    {
        let header = Self::parse_mtx_header(path.as_ref())?;
        let file_len = std::fs::metadata(path.as_ref())?.len();
        let (nrows, ncols, nnz_declared) = (header.nodes, header.nodes, header.entries);
        let header_offset = header.data_offset;
        // mirrored entries: each stands for an arc both ways
        let symmetric = header.symmetry.mirrored();
        // labels are non-negative integers: they can hold integer values, or none
        let pattern = match (header.field, header.symmetry) {
            (MtxField::Pattern, _) => true,
            (MtxField::Real | MtxField::Complex, _) => {
                return Err(format!(
                    "error edge labels are integers, so a MatrixMarket file of {} values can't be loaded with them: load it with unlabelled edges",
                    header.field
                )
                .into());
            }
            (MtxField::Integer, MtxSymmetry::SkewSymmetric) => {
                return Err("error edge labels are non-negative integers, so a skew-symmetric MatrixMarket file (whose mirrored values are negated) can't be loaded with them: load it with unlabelled edges".into());
            }
            (MtxField::Integer, _) => false,
        };
        let thread_load = file_len
            .saturating_sub(header_offset)
            .div_ceil(threads as u64);

        let seen = Arc::new(AtomicUsize::new(0));
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::new();

            for tid in 0..threads {
                let begin_pos = std::cmp::min(header_offset + tid as u64 * thread_load, file_len);
                let end_pos =
                    std::cmp::min(header_offset + (tid + 1) as u64 * thread_load, file_len)
                        as usize;
                let p = path.as_ref();

                let seen = seen.clone();
                let mut emit = emit.clone();

                let handle = s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        // open the file separately for each thread
                        let file = File::open(p)?;
                        let mut cur_pos = begin_pos as usize;

                        // line up bufreader to next line start
                        let mut rdr = BufReader::new(file);
                        if tid == 0 {
                            rdr.seek(SeekFrom::Start(begin_pos))?;
                        } else {
                            // this way we avoid the case where thread's bufreader startsoff
                            // aligned and the first edge is skipped
                            rdr.seek(SeekFrom::Start(begin_pos.saturating_sub(1)))?;
                            cur_pos += rdr.skip_until(b'\n')?.saturating_sub(1);
                        }

                        let mut line = String::new();

                        // the whole byte range, whatever the header declares: both passes over the
                        // file must read the same entries (the total is checked against it after)
                        while end_pos > cur_pos {
                            line.clear();

                            let n = rdr.read_line(&mut line)?;
                            if n == 0 {
                                break;
                            } else {
                                cur_pos += n;
                            }

                            let t = line.trim();
                            if t.is_empty() || t.starts_with('%') {
                                continue;
                            }

                            let mut it = t.split_whitespace();

                            // 1st two tokens must be i j (1-based)
                            let i1: usize = match it.next() {
                                Some(s) => s.parse()?,
                                None => continue, // malformed line; skip
                            };

                            let j1: usize = match it.next() {
                                Some(s) => s.parse()?,
                                None => continue, // malformed line; skip
                            };

                            // Convert to 0-based, also ensure in-bounds
                            if i1 == 0 || j1 == 0 {
                                return Err(format!(
                                    "MatrixMarket indices are 1-based; found 0 ({line})"
                                )
                                .into());
                            }

                            let i = i1 - 1;
                            let j = j1 - 1;

                            let w: usize = if pattern {
                                0
                            } else {
                                match it.next() {
                                    Some(s) => s.parse().map_err(|_| {
                                        format!("error entry {line:?}'s value {s:?} isn't a non-negative integer, which edge labels are")
                                    })?,
                                    None => 0, // malformed line; skip
                                }
                            };

                            if i >= nrows || j >= ncols {
                                return Err(format!(
                                    "Entry out of bounds: ({i},{j}) not in [{nrows}x{ncols}]"
                                )
                                .into());
                            }

                            // Emit edges:
                            // first directed edge i --(w)--> j
                            emit(i, j, w);

                            // second directed edge (if symmetrical) j --(w)--> i
                            if symmetric && i != j {
                                emit(j, i, w);
                            }

                            seen.add(1, Ordering::Relaxed);
                        }
                        Ok(())
                    },
                );

                handles.push(handle);
            }

            for (idx, handle) in handles.into_iter().enumerate() {
                handle
                    .join()
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error joining thread {idx}: {:?}", e).into()
                    })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in thread {idx}: {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        let entries = seen.load(Ordering::Relaxed);
        if entries != nnz_declared {
            return Err(format!(
                "error the MatrixMarket file declares {nnz_declared} entries, but has {entries}"
            )
            .into());
        }

        Ok(())
    }

    #[deprecated]
    fn parse_mtx_with<F, P: AsRef<Path>>(
        path: P,
        mut emit: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, u64, u64),
    {
        let f = File::open(path.as_ref())?;
        let mut rdr = BufReader::new(f);

        // read and parse the header line
        let mut header = String::new();
        rdr.read_line(&mut header)?;
        if !header.starts_with("%%MatrixMarket") {
            return Err("Invalid MatrixMarket header".into());
        }
        // tokens: 0=%%MatrixMarket, 1=matrix, 2=coordinate, 3=field, 4=symmetry
        let toks: Vec<_> = header.split_whitespace().collect();
        if toks.len() < 5 {
            return Err("Malformed MatrixMarket header".into());
        }
        if !toks[1].eq_ignore_ascii_case("matrix")
            || !toks[2].eq_ignore_ascii_case("coordinate")
            || !(toks[3].eq_ignore_ascii_case("pattern") || toks[3].eq_ignore_ascii_case("integer"))
            || !(toks[4].eq_ignore_ascii_case("symmetric")
                || toks[4].eq_ignore_ascii_case("skew-symmetric")
                || toks[4].eq_ignore_ascii_case("general"))
        {
            return Err(
                "Only 'matrix coordinate pattern/integer symmetric' format is supported".into(),
            );
        }
        let pattern = toks[3].eq_ignore_ascii_case("pattern");
        let symmetric = !toks[4].eq_ignore_ascii_case("general");

        // skip comment lines (%) to find the size line: "nrows ncols nnz"
        let (nrows, ncols, nnz_declared) = {
            let mut line = String::new();
            loop {
                line.clear();
                let n = rdr.read_line(&mut line)?;
                if n == 0 {
                    return Err("Unexpected EOF before reading size line".into());
                }
                // ignore comment lines
                let line_trim = line.trim();
                if line_trim.is_empty() || line_trim.starts_with('%') {
                    continue;
                }
                // parse sizes
                let mut it = line_trim.split_whitespace();
                let nr: usize = it.next().ok_or("Missing nrows")?.parse()?;
                let nc: usize = it.next().ok_or("Missing ncols")?.parse()?;
                let nnz: usize = it.next().ok_or("Missing nnz")?.parse()?;
                if nr != nc {
                    return Err(format!(
                        "Only symmetric matrices are supported but {{nr = {nr}}} != {{nc = {nc}}}"
                    )
                    .into());
                }
                break (nr, nc, nnz);
            }
        };

        // stream the nnz entries
        // For coordinate:
        // - pattern:          i j
        // - integer/real:     i j val
        // - complex/hermitian i j real imag  (we’ll ignore values and just read i j)
        //
        // MatrixMarket is 1-based; convert to 0-based.
        // We’ll be forgiving: we keep reading non-comment non-empty lines until we see nnz_declared entries.
        let mut seen = 0usize;
        let mut line = String::new();
        while seen < nnz_declared {
            line.clear();
            let n = rdr.read_line(&mut line)?;
            if n == 0 {
                // Some files under-report nnz in the header; we stop if EOF reached
                break;
            }
            let t = line.trim();
            if t.is_empty() || t.starts_with('%') {
                continue;
            }
            let mut it = t.split_whitespace();

            // 1st two tokens must be i j (1-based)
            let i1: usize = match it.next() {
                Some(s) => s.parse()?,
                None => continue, // malformed line; skip
            };
            let j1: u64 = match it.next() {
                Some(s) => s.parse()?,
                None => continue, // malformed line; skip
            };
            // Convert to 0-based, also ensure in-bounds
            if i1 == 0 || j1 == 0 {
                return Err("MatrixMarket indices are 1-based; found 0".into());
            }
            let i = i1 - 1;
            let j = j1 - 1;

            let w: u64 = if pattern {
                0
            } else {
                match it.next() {
                    Some(s) => s.parse()?,
                    None => 0, // malformed line; skip
                }
            };
            if i >= nrows || j as usize >= ncols {
                return Err(
                    format!("Entry out of bounds: ({i},{j}) not in [{nrows}x{ncols}]").into(),
                );
            }

            // Emit edges:
            // one directed edge i --(w)--> j
            emit(i, j, w);
            if symmetric && i != j as usize {
                emit(j as usize, i as u64, w);
            }

            seen += 1;
        }

        Ok(())
    }
}
