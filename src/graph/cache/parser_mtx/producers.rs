use std::{
    fs::File,
    io::{BufRead, BufReader, Seek},
    path::Path,
};

use crossbeam::thread;
use portable_atomic::{AtomicUsize, Ordering};

use super::{GraphCache, MtxField, MtxHeader, MtxSymmetry};

/// Whitespace within a MatrixMarket line.
#[inline(always)]
fn is_blank(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | 0x0b | 0x0c)
}

/// Splits the next whitespace-separated token off `rest`.
#[inline(always)]
fn next_token<'a>(rest: &mut &'a [u8]) -> Option<&'a [u8]> {
    let from = rest.iter().position(|&b| !is_blank(b))?;
    let len = rest[from..]
        .iter()
        .position(|&b| is_blank(b))
        .unwrap_or(rest.len() - from);
    let tok = &rest[from..from + len];
    *rest = &rest[from + len..];
    Some(tok)
}

/// The first token of `rest`, or nothing if it has none.
#[inline(always)]
fn first_token(mut rest: &[u8]) -> &[u8] {
    next_token(&mut rest).unwrap_or(b"")
}

/// Parses a (non-empty) decimal index, failing on any non-digit or on overflow.
#[inline(always)]
fn parse_index(tok: &[u8]) -> Option<usize> {
    let mut acc = 0usize;
    for &b in tok {
        let d = b.wrapping_sub(b'0');
        if d > 9 {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_add(d as usize)?;
    }
    Some(acc)
}

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

    /// Parses every entry into `emit(i, j)` (0-based), mirrored entries both ways, returning how
    /// many arcs were emitted.
    pub(super) fn parallel_no_labels_parse_mtx_with<F, P: AsRef<Path>>(
        path: P,
        emit: F,
        threads: usize,
    ) -> Result<usize, Box<dyn std::error::Error>>
    where
        F: FnMut(usize, usize) + Send + Sync + Clone,
    {
        Self::parallel_scan_mtx_entries(path.as_ref(), threads, {
            let mut emit = emit;
            move |i, j, _, mirrored| {
                emit(i, j);
                if mirrored {
                    emit(j, i);
                }
                Ok(())
            }
        })
    }

    /// Parses every entry into `emit(i, j, label)` (0-based), mirrored entries both ways. An
    /// entry's label is `value(token)` of its value token (empty if missing), or `unit` for
    /// `pattern` files, which have none; the reverse arc of a skew-symmetric entry gets the
    /// label's [`negate_weight`](crate::graph::E::negate_weight). Which values `E` can hold must
    /// already have been checked against the header (see `check_mtx_values`).
    pub(super) fn parallel_edge_labels_parse_mtx_with<F, V, P: AsRef<Path>>(
        path: P,
        emit: F,
        value: V,
        unit: E,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, usize, E) + Send + Sync + Clone,
        V: Fn(&[u8]) -> Result<E, Box<dyn std::error::Error + Send + Sync>> + Sync,
    {
        let header = Self::parse_mtx_header(path.as_ref())?;
        let pattern = header.field == MtxField::Pattern;
        let skew = header.symmetry == MtxSymmetry::SkewSymmetric;
        let value = &value;
        Self::parallel_scan_mtx_entries(path.as_ref(), threads, {
            let mut emit = emit;
            move |i, j, rest, mirrored| {
                let w = if pattern {
                    unit
                } else {
                    value(first_token(rest))?
                };
                emit(i, j, w);
                if mirrored {
                    emit(j, i, if skew { w.negate_weight() } else { w });
                }
                Ok(())
            }
        })?;
        Ok(())
    }

    /// The scanner both passes share: memory maps the file, splits its body into one byte range
    /// per thread (a line belongs to the range it starts in) and, for each entry `i j rest...`,
    /// calls `entry(i, j, rest, mirrored)` with 0-based, bounds-checked `i` and `j`, the bytes
    /// after them, and whether the entry also stands for the arc `j -> i`. Blank lines, comments
    /// and lines of fewer than two tokens are skipped. Returns the number of arcs the entries
    /// stand for, after checking their number against the header's.
    ///
    /// Works on raw bytes: no line buffers, no UTF-8 validation, indices parsed by a digit loop
    /// (still validated: they index memory), and no shared counters touched per entry.
    fn parallel_scan_mtx_entries<F>(
        path: &Path,
        threads: usize,
        entry: F,
    ) -> Result<usize, Box<dyn std::error::Error>>
    where
        F: FnMut(usize, usize, &[u8], bool) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Clone,
    {
        let header = Self::parse_mtx_header(path)?;
        let (nodes, nnz_declared) = (header.nodes, header.entries);
        // mirrored entries: each (off-diagonal one) stands for an arc both ways
        let symmetric = header.symmetry.mirrored();
        let file = File::open(path)?;
        // SAFETY: read only, and the input isn't expected to change while it's parsed (as for
        // every memory mapped file of the cache)
        let data = unsafe { memmap2::Mmap::map(&file)? };
        let data: &[u8] = &data;
        let body = (header.data_offset as usize).min(data.len());
        let load = (data.len() - body).div_ceil(threads.max(1)).max(1);

        let (seen, arcs) = (AtomicUsize::new(0), AtomicUsize::new(0));
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            for tid in 0..threads {
                let begin = (body + tid * load).min(data.len());
                let end = (body + (tid + 1) * load).min(data.len());
                let (seen, arcs) = (&seen, &arcs);
                let mut entry = entry.clone();
                handles.push(s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        // the first line starting at or after `begin` (the body's first is at it)
                        let mut start = if tid == 0 {
                            begin
                        } else {
                            memchr::memchr(b'\n', &data[begin - 1..])
                                .map_or(data.len(), |p| begin + p)
                        };
                        let (mut local_seen, mut local_arcs) = (0usize, 0usize);
                        while start < end {
                            let line_end = memchr::memchr(b'\n', &data[start..])
                                .map_or(data.len(), |p| start + p);
                            let line = &data[start..line_end];
                            start = line_end + 1;

                            let mut rest = line;
                            let Some(i1) = next_token(&mut rest) else {
                                continue; // blank
                            };
                            if i1[0] == b'%' {
                                continue; // comment
                            }
                            let Some(j1) = next_token(&mut rest) else {
                                continue; // malformed: skipped (and then miscounted)
                            };
                            let (i, j) = match (parse_index(i1), parse_index(j1)) {
                                (Some(i1), Some(j1)) if i1 > 0 && j1 > 0 => (i1 - 1, j1 - 1),
                                (Some(0), _) | (_, Some(0)) => {
                                    return Err(format!(
                                        "MatrixMarket indices are 1-based; found 0 ({:?})",
                                        String::from_utf8_lossy(line).trim()
                                    )
                                    .into());
                                }
                                _ => {
                                    return Err(format!(
                                        "error MatrixMarket entry {:?} has a malformed index",
                                        String::from_utf8_lossy(line).trim()
                                    )
                                    .into());
                                }
                            };
                            if i >= nodes || j >= nodes {
                                return Err(format!(
                                    "Entry out of bounds: ({i},{j}) not in [{nodes}x{nodes}]"
                                )
                                .into());
                            }
                            let mirrored = symmetric && i != j;
                            entry(i, j, rest, mirrored).map_err(|e| {
                                format!(
                                    "error in entry {:?}: {e}",
                                    String::from_utf8_lossy(line).trim()
                                )
                            })?;
                            local_seen += 1;
                            local_arcs += 1 + mirrored as usize;
                        }
                        seen.add(local_seen, Ordering::Relaxed);
                        arcs.add(local_arcs, Ordering::Relaxed);
                        Ok(())
                    },
                ));
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

        // the whole byte range was read, whatever the header declares: both passes over the file
        // read the same entries, and their total must be the declared one
        let entries = seen.load(Ordering::Relaxed);
        if entries != nnz_declared {
            return Err(format!(
                "error the MatrixMarket file declares {nnz_declared} entries, but has {entries}"
            )
            .into());
        }
        Ok(arcs.load(Ordering::Relaxed))
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
