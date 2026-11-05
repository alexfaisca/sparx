use crate::graph::cache::{
    GraphCache, MultithreadedParserIndexBounds, utils::apply_permutation_in_place,
};

use crossbeam::thread;

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> GraphCache<N, E, Ix> {
    /// Parses a ggcat output edge direction
    ///
    /// # Arguments
    ///
    /// * `orig` - The edge's origin endpoint annotation. Must be either '+' or '-'.
    /// * `destiny` - The edge's destiny endpoint annotation. Must be either '+' or '-'.
    ///
    /// # Returns
    ///
    /// A `usize` that encodes the edge direction:
    /// * `0usize` --- for a forward-forward edge.
    /// * `1usize` --- for a forward-reverse edge.
    /// * `2usize` --- for a reverse-forward edge.
    /// * `3usize` --- for a reverse-reverse edge.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * An unknown annotation is given as input.
    ///
    pub(super) fn parse_ggcat_direction(
        orig: &str,
        dest: &str,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        match (orig, dest) {
            ("+", "+") => Ok(0),
            ("+", "-") => Ok(1),
            ("-", "+") => Ok(2),
            ("-", "-") => Ok(3),
            _ => Err(format!("error ubknown edge direction annotations (supported are '+' && '-'): EdgeAnnotations {{orig: '{orig}', dest: '{dest}'}}").into()),
        }
    }

    #[inline]
    fn parse_usize_decimal(bytes: &[u8]) -> Option<usize> {
        let mut x: usize = 0;
        for &b in bytes {
            // stop at first non-digit (space, newline, ',', ':', etc.)
            if !b.is_ascii_digit() {
                break;
            }
            let d = (b - b'0') as usize;
            x = x.saturating_mul(10).saturating_add(d);
        }
        Some(x)
    }

    /// Parses `C:53:2` -> returns the **last** number (e.g., color class id = 2).
    /// Works for `C:x:y` or `C:y` by picking the last numeric run.
    ///
    /// Not supposed to be permanent, just to test nodelabels.
    #[inline]
    fn parse_c_colon(line: &str) -> Option<usize> {
        let s = line.as_bytes();
        // scan from end to find the last contiguous digit run
        let mut end = s.len();
        while end > 0 && !(s[end - 1].is_ascii_digit()) {
            end -= 1;
        }
        let mut start = end;
        while start > 0 && s[start - 1].is_ascii_digit() {
            start -= 1;
        }
        if start == end {
            return None;
        }
        Self::parse_usize_decimal(&s[start..end])
    }

    fn parse_ggcat_builder_thread_bounds(
        input: &[u8],
        threads: usize,
    ) -> Result<MultithreadedParserIndexBounds, Box<dyn std::error::Error>> {
        let thread_load = input.len().div_ceil(threads);

        // figure out thread bounds
        let mut bounds = vec![(0usize, 0usize); threads];
        let mut previous_end = 0usize;
        (0..threads).try_for_each(|tid| -> Result<(), Box<dyn std::error::Error>> {
            let begin = previous_end;
            if begin != input.len() && input[begin] != b'>' {
                return Err(format!("error getting threads bounds for thread {tid}: input[{{thread_begin: {begin}}}] = {} (should equal '>')", input[begin]).into());
            }
            let mut end = std::cmp::min((tid + 1) * thread_load, input.len());

            // find beginning of next node entry after end of slice (marked by '>')
            while end < input.len() && input[end] != b'>' {
                end += 1;
            }
            previous_end = end;
            bounds[tid] = (begin, end);
            Ok(())
        })?;

        Ok(bounds.into_boxed_slice())
    }

    pub(super) fn parse_ggcat_max_node_id(
        input: &[u8],
    ) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        let mut begin = input.len().saturating_sub(1);
        let mut last_space = begin;

        while begin > 0 && input[begin] != b'>' {
            if input[begin] == b' ' {
                last_space = begin;
            }
            begin -= 1;
        }

        // only possible if input.len() == 0 or no '>' markers exist on the file
        // in either case, file is considered empty
        if begin == input.len() || begin == 0 && input[begin] != b'>' {
            return Ok(None);
        }

        let it = input[begin + 1..last_space].iter().copied();
        let mut acc: usize = 0;
        let mut saw_digit = false;
        for b in it {
            if !b.is_ascii_digit() {
                return Err(format!(
                    "error parsing ggcat input file's last node index: {b} is not a valid digit"
                )
                .into());
            }
            saw_digit = true;
            let d = (b - b'0') as usize;
            acc = acc
                .checked_mul(10)
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    format!(
                        "error parsing ggcat input file's last node index: {acc} * 10 overflowed"
                    )
                    .into()
                })?;
            acc = acc
                .checked_add(d)
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    format!(
                        "error parsing ggcat input file's last node index: {acc} + {d} overflowed"
                    )
                    .into()
                })?;
        }
        // get max node id
        if !saw_digit {
            return Err("error parsing ggcat input file's last node index: not one valid ascii digit was found".into());
        }

        Ok(Some(acc))
    }

    pub(super) fn parallel_edge_counter_parse_ggcat_bytes_mmap_with<F>(
        input: &[u8],
        emit: F,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, usize) + Send + Sync + Clone,
    {
        let bounds = Self::parse_ggcat_builder_thread_bounds(input, threads)?;

        // write edges
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let mut emit = emit.clone();

                handles.push(s.spawn(move |_|  -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    // this assumes UTF-8 but avoids full conversion 
                    let mut lines = input.split(|&b| b == b'\n');

                    while let Some(line) = lines.next() {
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;
                        let _label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error in thread {tid}: no k-mer sequence for node {line_str}").into()
                        })?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;
                        let _node_lengh = node.next(); // length 
                        let _node_color = node.next(); // color value

                        emit(id, node.len());
                    }

                    // return slice with all the tmp fst filenames
                    Ok(())
                }));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting node degrees from ggcat file (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        Ok(())
    }

    pub(super) fn parallel_no_labels_parse_ggcat_bytes_mmap_with<F>(
        input: &[u8],
        emit: F,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, &[Ix]) + Send + Sync + Clone,
    {
        let bounds = Self::parse_ggcat_builder_thread_bounds(input, threads)?;

        // write edges
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let mut emit = emit.clone();

                handles.push(s.spawn(move |_|  -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    // this assumes UTF-8 but avoids full conversion 
                    let mut lines = input.split(|&b| b == b'\n');
                    let mut node_neighbors: Vec<Ix> = Vec::with_capacity(u16::MAX as usize);

                    while let Some(line) = lines.next() {
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;
                        let _label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error in thread {tid}: no k-mer sequence for node {line_str}").into()
                        })?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;
                        let _node_lengh = node.next(); // length 
                        let _node_color = node.next(); // color value

                        for link in node {
                            let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
                            let dest: usize = link_slice[1].parse()?;
                            node_neighbors.push(Ix::new(dest));
                        }

                        node_neighbors.sort_unstable_by_key(|&n| n);

                        emit(id, node_neighbors.as_slice());
                        node_neighbors.clear();
                    }

                    // return slice with all the tmp fst filenames
                    Ok(())
                }));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting node degrees from ggcat file (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        Ok(())
    }

    pub(super) fn parallel_node_labels_parse_ggcat_bytes_mmap_with<F>(
        input: &[u8],
        emit: F,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, &[Ix], N) + Send + Sync + Clone,
    {
        let bounds = Self::parse_ggcat_builder_thread_bounds(input, threads)?;

        // write edges
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let mut emit = emit.clone();

                handles.push(s.spawn(move |_|  -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    // this assumes UTF-8 but avoids full conversion 
                    let mut lines = input.split(|&b| b == b'\n');
                    let mut node_neighbors: Vec<Ix> = Vec::with_capacity(u16::MAX as usize);

                    while let Some(line) = lines.next() {
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;
                        let _label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error in thread {tid}: no k-mer sequence for node {line_str}").into()
                        })?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;
                        let _node_lengh = node.next(); // length 
                        let node_color = node.next(); // color value
                                                      //
                        let color = if N::is_labeled() {
                            N::new(Self::parse_c_colon(node_color.unwrap()).unwrap_or(0))
                        } else {
                            N::new(0)
                        };

                        for link in node {
                            let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
                            let dest: usize = link_slice[1].parse()?;
                            node_neighbors.push(Ix::new(dest));
                        }

                        node_neighbors.sort_unstable_by_key(|e| *e);

                        emit(id, node_neighbors.as_slice(), color);
                        node_neighbors.clear();
                    }

                    // return slice with all the tmp fst filenames
                    Ok(())
                }));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting node degrees from ggcat file (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        Ok(())
    }

    pub(super) fn parallel_edge_labels_parse_ggcat_bytes_mmap_with<F>(
        input: &[u8],
        emit: F,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, &[Ix], &[E]) + Send + Sync + Clone,
    {
        let bounds = Self::parse_ggcat_builder_thread_bounds(input, threads)?;

        // write edges
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let mut emit = emit.clone();

                handles.push(s.spawn(move |_|  -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    // this assumes UTF-8 but avoids full conversion 
                    let mut lines = input.split(|&b| b == b'\n');
                    let mut node_neighbors: Vec<Ix> = Vec::with_capacity(u16::MAX as usize);
                    let mut edge_labels = Vec::with_capacity(u16::MAX as usize);
                    let mut idx = Vec::with_capacity(u16::MAX as usize);

                    while let Some(line) = lines.next() {
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;
                        let _label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error in thread {tid}: no k-mer sequence for node {line_str}").into()
                        })?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;
                        let _node_lengh = node.next(); // length 
                        let _node_color = node.next(); // color value

                        for link in node {
                            let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
                            let dest: usize = link_slice[1].parse()?;
                            node_neighbors.push(Ix::new(dest));
                            idx.push(idx.len());
                            if E::is_labeled() {
                                edge_labels.push(
                                    E::new(Self::parse_ggcat_direction(
                                        link_slice[0],
                                        link_slice[2])
                                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error in thread {tid}: {e}").into()})?,
                                    ));
                            } else {
                                edge_labels.push(E::new(0));
                            }
                        }

                        // FIXME: test sort e_labels
                        idx.sort_by_key(|&i| node_neighbors[i]);
                        apply_permutation_in_place(idx.as_mut_slice(), node_neighbors.as_mut_slice(), edge_labels.as_mut_slice());

                        emit(id, node_neighbors.as_slice(), edge_labels.as_slice());
                        node_neighbors.clear();
                        edge_labels.clear();
                        idx.clear();
                    }

                    // return slice with all the tmp fst filenames
                    Ok(())
                }));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting node degrees from ggcat file (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        Ok(())
    }

    pub(super) fn parallel_node_edge_labels_parse_ggcat_bytes_mmap_with<F>(
        input: &[u8],
        emit: F,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, &[Ix], N, &[E]) + Send + Sync + Clone,
    {
        let bounds = Self::parse_ggcat_builder_thread_bounds(input, threads)?;

        // write edges
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let mut emit = emit.clone();

                handles.push(s.spawn(move |_|  -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    // this assumes UTF-8 but avoids full conversion 
                    let mut lines = input.split(|&b| b == b'\n');
                    let mut node_neighbors: Vec<Ix> = Vec::with_capacity(u16::MAX as usize);
                    let mut edge_labels = Vec::with_capacity(u16::MAX as usize);
                    let mut idx = Vec::with_capacity(u16::MAX as usize);

                    while let Some(line) = lines.next() {
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;
                        let _label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error in thread {tid}: no k-mer sequence for node {line_str}").into()
                        })?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;
                        let _node_lengh = node.next(); // length 
                        let node_color = node.next(); // color value
                                                      //
                        let color = if N::is_labeled() {
                            N::new(Self::parse_c_colon(node_color.unwrap()).unwrap_or(0))
                        } else {
                            N::new(0)
                        };

                        for link in node {
                            let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
                            let dest: usize = link_slice[1].parse()?;
                            node_neighbors.push(Ix::new(dest));
                            idx.push(idx.len());
                            if E::is_labeled() {
                                edge_labels.push(
                                    E::new(Self::parse_ggcat_direction(
                                        link_slice[0],
                                        link_slice[2])
                                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error in thread {tid}: {e}").into()})?,
                                    ));
                            } else {
                                edge_labels.push(E::new(0));
                            }
                        }

                        // FIXME: test sort e_labels
                        idx.sort_by_key(|&i| node_neighbors[i]);
                        apply_permutation_in_place(idx.as_mut_slice(), node_neighbors.as_mut_slice(), edge_labels.as_mut_slice());

                        emit(id, node_neighbors.as_slice(), color, edge_labels.as_slice());
                        node_neighbors.clear();
                        edge_labels.clear();
                        idx.clear();
                    }

                    // return slice with all the tmp fst filenames
                    Ok(())
                }));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting node degrees from ggcat file (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        Ok(())
    }

    pub(super) fn parallel_meta_labels_parse_ggcat_bytes_mmap_with<F, FF, FB>(
        input: &[u8],
        emit: F,
        fst_filter: FF,
        emit_batch: FB,
        batch_size: usize,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, &[Ix]) + Send + Sync + Clone,
        FF: FnMut(usize) -> bool + Send + Sync + Clone,
        FB: FnMut(&mut [(&[u8], u64)]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Sync
            + Clone,
    {
        let bounds = Self::parse_ggcat_builder_thread_bounds(input, threads)?;

        // write edges
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let mut emit = emit.clone();
                let mut in_fst = fst_filter.clone();
                let mut emit_batch = emit_batch.clone();

                let mut current_batch: Vec<(&[u8], u64)> = Vec::with_capacity(batch_size);

                handles.push(s.spawn(move |_|  -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    // this assumes UTF-8 but avoids full conversion 
                    let mut lines = input.split(|&b| b == b'\n');
                    let mut node_neighbors: Vec<Ix> = Vec::with_capacity(u16::MAX as usize);

                    while let Some(line) = lines.next() {
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;
                        let label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error in thread {tid}: no k-mer sequence for node {line_str}").into()
                        })?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;
                        let _node_lengh = node.next(); // length 
                        let _node_color = node.next(); // color value

                        for link in node {
                            let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
                            let dest: usize = link_slice[1].parse()?;
                            node_neighbors.push(Ix::new(dest));
                        }

                        if in_fst(id) {
                            current_batch.push((label_line, id as u64));
                            if current_batch.len() >= batch_size {
                                // println!("wrote {}", batch_num.load(Ordering::Relaxed));
                                emit_batch(current_batch.as_mut_slice())?;
                                current_batch.clear();
                            }
                        }

                        node_neighbors.sort_unstable_by_key(|e| *e);

                        emit(id, node_neighbors.as_slice());
                        node_neighbors.clear();
                    }

                    // emit the last batch if not empty
                    if !current_batch.is_empty() {
                        emit_batch(current_batch.as_mut_slice())?;
                    }

                    // return slice with all the tmp fst filenames
                    Ok(())
                }));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting node degrees from ggcat file (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        Ok(())
    }

    pub(super) fn parallel_edge_meta_labels_parse_ggcat_bytes_mmap_with<F, FF, FB>(
        input: &[u8],
        emit: F,
        fst_filter: FF,
        emit_batch: FB,
        batch_size: usize,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, &[Ix], &[E]) + Send + Sync + Clone,
        FF: FnMut(usize) -> bool + Send + Sync + Clone,
        FB: FnMut(&mut [(&[u8], u64)]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Sync
            + Clone,
    {
        let bounds = Self::parse_ggcat_builder_thread_bounds(input, threads)?;

        // write edges
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let mut emit = emit.clone();
                let mut in_fst = fst_filter.clone();
                let mut emit_batch = emit_batch.clone();

                handles.push(s.spawn(move |_|  -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    // this assumes UTF-8 but avoids full conversion 
                    let mut lines = input.split(|&b| b == b'\n');
                    let mut node_neighbors: Vec<Ix> = Vec::with_capacity(u16::MAX as usize);
                    let mut edge_labels = Vec::with_capacity(u16::MAX as usize);
                    let mut idx = Vec::with_capacity(u16::MAX as usize);

                    let mut current_batch: Vec<(&[u8], u64)> = Vec::with_capacity(batch_size);

                    while let Some(line) = lines.next() {
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;
                        let label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error in thread {tid}: no k-mer sequence for node {line_str}").into()
                        })?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;
                        let _node_lengh = node.next(); // length 
                        let _node_color = node.next(); // color value

                        for link in node {
                            let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
                            let dest: usize = link_slice[1].parse()?;
                            node_neighbors.push(Ix::new(dest));
                            idx.push(idx.len());
                            if E::is_labeled() {
                                edge_labels.push(
                                    E::new(Self::parse_ggcat_direction(
                                        link_slice[0],
                                        link_slice[2])
                                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error in thread {tid}: {e}").into()})?,
                                    ));
                            } else {
                                edge_labels.push(E::new(0));
                            }
                        }

                        if in_fst(id) {
                            current_batch.push((label_line, id as u64));
                            if current_batch.len() >= batch_size {
                                // println!("wrote {}", batch_num.load(Ordering::Relaxed));
                                emit_batch(current_batch.as_mut_slice())?;
                                current_batch.clear();
                            }
                        }

                        // FIXME: test sort e_labels
                        idx.sort_by_key(|&i| node_neighbors[i]);
                        apply_permutation_in_place(idx.as_mut_slice(), node_neighbors.as_mut_slice(), edge_labels.as_mut_slice());

                        emit(id, node_neighbors.as_slice(), edge_labels.as_slice());
                        node_neighbors.clear();
                        edge_labels.clear();
                        idx.clear();
                    }

                    // emit the last batch if not empty
                    if !current_batch.is_empty() {
                        emit_batch(current_batch.as_mut_slice())?;
                    }

                    // return slice with all the tmp fst filenames
                    Ok(())
                }));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting node degrees from ggcat file (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        Ok(())
    }

    pub(super) fn parallel_node_meta_labels_parse_ggcat_bytes_mmap_with<F, FF, FB>(
        input: &[u8],
        emit: F,
        fst_filter: FF,
        emit_batch: FB,
        batch_size: usize,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, &[Ix], N) + Send + Sync + Clone,
        FF: FnMut(usize) -> bool + Send + Sync + Clone,
        FB: FnMut(&mut [(&[u8], u64)]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Sync
            + Clone,
    {
        let bounds = Self::parse_ggcat_builder_thread_bounds(input, threads)?;

        // write edges
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let mut emit = emit.clone();
                let mut in_fst = fst_filter.clone();
                let mut emit_batch = emit_batch.clone();

                handles.push(s.spawn(move |_|  -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    // this assumes UTF-8 but avoids full conversion 
                    let mut lines = input.split(|&b| b == b'\n');
                    let mut node_neighbors: Vec<Ix> = Vec::with_capacity(u16::MAX as usize);

                    let mut current_batch: Vec<(&[u8], u64)> = Vec::with_capacity(batch_size);

                    while let Some(line) = lines.next() {
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;
                        let label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error in thread {tid}: no k-mer sequence for node {line_str}").into()
                        })?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;
                        let _node_lengh = node.next(); // length 
                        let node_color = node.next(); // color value
                                                      //
                        let color = if N::is_labeled() {
                            N::new(Self::parse_c_colon(node_color.unwrap()).unwrap_or(0))
                        } else {
                            N::new(0)
                        };

                        for link in node {
                            let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
                            let dest: usize = link_slice[1].parse()?;
                            node_neighbors.push(Ix::new(dest));
                        }

                        if in_fst(id) {
                            current_batch.push((label_line, id as u64));
                            if current_batch.len() >= batch_size {
                                // println!("wrote {}", batch_num.load(Ordering::Relaxed));
                                emit_batch(current_batch.as_mut_slice())?;
                                current_batch.clear();
                            }
                        }

                        node_neighbors.sort_by_key(|e| *e);

                        emit(id, node_neighbors.as_slice(), color);
                        node_neighbors.clear();
                    }

                    // emit the last batch if not empty
                    if !current_batch.is_empty() {
                        emit_batch(current_batch.as_mut_slice())?;
                    }

                    // return slice with all the tmp fst filenames
                    Ok(())
                }));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting node degrees from ggcat file (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        Ok(())
    }

    pub(super) fn parallel_full_labels_parse_ggcat_bytes_mmap_with<'a, F, FF, FB>(
        input: &'a [u8],
        emit: F,
        fst_filter: FF,
        emit_batch: FB,
        batch_size: usize,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, &[Ix], N, &[E]) + Send + Sync + Clone,
        FF: FnMut(usize) -> bool + Send + Sync + Clone,
        FB: for<'b> FnMut(
                &'b mut [(&'a [u8], u64)],
            ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Sync
            + Clone,
    {
        let bounds = Self::parse_ggcat_builder_thread_bounds(input, threads)?;

        // write edges
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let mut emit = emit.clone();
                let mut in_fst = fst_filter.clone();
                let mut emit_batch = emit_batch.clone();

                handles.push(s.spawn(move |_|  -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    // this assumes UTF-8 but avoids full conversion 
                    let mut lines = input.split(|&b| b == b'\n');
                    let mut node_neighbors: Vec<Ix> = Vec::with_capacity(u16::MAX as usize);
                    let mut edge_labels = Vec::with_capacity(u16::MAX as usize);
                    let mut idx = Vec::with_capacity(u16::MAX as usize);

                    let mut current_batch: Vec<(&'a [u8], u64)> = Vec::with_capacity(batch_size);

                    while let Some(line) = lines.next() {
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;
                        let label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error in thread {tid}: no k-mer sequence for node {line_str}").into()
                        })?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;
                        let _node_lengh = node.next(); // length 
                        let node_color = node.next(); // color value
                                                      //
                        let color = if N::is_labeled() {
                            N::new(Self::parse_c_colon(node_color.unwrap()).unwrap_or(0))
                        } else {
                            N::new(0)
                        };

                        for link in node {
                            let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
                            let dest: usize = link_slice[1].parse()?;
                            node_neighbors.push(Ix::new(dest));
                            idx.push(idx.len());
                            if E::is_labeled() {
                                edge_labels.push(
                                    E::new(Self::parse_ggcat_direction(
                                        link_slice[0],
                                        link_slice[2])
                                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error in thread {tid}: {e}").into()})?,
                                    ));
                            } else {
                                edge_labels.push(E::new(0));
                            }
                        }

                        if in_fst(id) {
                            current_batch.push((label_line, id as u64));
                            if current_batch.len() >= batch_size {
                                // println!("wrote {}", batch_num.load(Ordering::Relaxed));
                                emit_batch(current_batch.as_mut_slice())?;
                                current_batch.clear();
                            }
                        }

                        // FIXME: test sort e_labels
                        idx.sort_by_key(|&i| node_neighbors[i]);
                        apply_permutation_in_place(idx.as_mut_slice(), node_neighbors.as_mut_slice(), edge_labels.as_mut_slice());

                        emit(id, node_neighbors.as_slice(), color, edge_labels.as_slice());
                        node_neighbors.clear();
                        edge_labels.clear();
                        idx.clear();
                    }

                    // emit the last batch if not empty
                    if !current_batch.is_empty() {
                        emit_batch(current_batch.as_mut_slice())?;
                    }

                    // return slice with all the tmp fst filenames
                    Ok(())
                }));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting node degrees from ggcat file (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        Ok(())
    }

    pub(super) fn parallel_only_meta_labels_parse_ggcat_bytes_mmap_with<FF, FB>(
        input: &[u8],
        fst_filter: FF,
        emit_batch: FB,
        batch_size: usize,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        FF: FnMut(usize) -> bool + Send + Sync + Clone,
        FB: FnMut(&mut [(&[u8], u64)]) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
            + Send
            + Sync
            + Clone,
    {
        let bounds = Self::parse_ggcat_builder_thread_bounds(input, threads)?;

        // write edges
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let mut in_fst = fst_filter.clone();
                let mut emit_batch = emit_batch.clone();

                let mut current_batch: Vec<(&[u8], u64)> = Vec::with_capacity(batch_size);

                handles.push(s.spawn(move |_|  -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    // this assumes UTF-8 but avoids full conversion 
                    let mut lines = input.split(|&b| b == b'\n');

                    while let Some(line) = lines.next() {
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;
                        let label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error in thread {tid}: no k-mer sequence for node {line_str}").into()
                        })?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;

                        if in_fst(id) {
                            current_batch.push((label_line, id as u64));
                            if current_batch.len() >= batch_size {
                                // println!("wrote {}", batch_num.load(Ordering::Relaxed));
                                emit_batch(current_batch.as_mut_slice())?;
                                current_batch.clear();
                            }
                        }
                    }

                    // emit the last batch if not empty
                    if !current_batch.is_empty() {
                        emit_batch(current_batch.as_mut_slice())?;
                    }

                    // return slice with all the tmp fst filenames
                    Ok(())
                }));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting node degrees from ggcat file (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        Ok(())
    }
}
