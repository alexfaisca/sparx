use memmap2::{Mmap, MmapMut, MmapOptions};
use std::{
    fmt::Debug,
    fs::{File, OpenOptions},
    mem::{ManuallyDrop, size_of},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AbstractedProceduralMemory<T> {
    slice: SharedSlice<T>,
    mmap: Arc<Mmap>,
    _vec: Vec<T>,
    mmapped: bool,
}

#[derive(Debug)]
pub struct AbstractedProceduralMemoryMut<T> {
    slice: SharedSliceMut<T>,
    mmap: MmapMut,
    _vec: Vec<T>,
    mmapped: bool,
}

#[derive(Copy, Debug)]
pub(crate) struct SharedSlice<T> {
    ptr: *const T,
    len: usize,
}

#[derive(Copy, Debug)]
pub struct SharedSliceMut<T> {
    pub ptr: *mut T,
    len: usize,
}

#[derive(Clone)]
pub(crate) struct SharedQueueMut<T> {
    pub ptr: *mut T,
    max: usize,
    read: Arc<AtomicUsize>,
    write: Arc<AtomicUsize>,
    len: Arc<AtomicUsize>,
}

unsafe impl<T> Send for SharedSlice<T> {}
unsafe impl<T> Sync for SharedSlice<T> {}

unsafe impl<T> Send for SharedSliceMut<T> {}
unsafe impl<T> Sync for SharedSliceMut<T> {}

unsafe impl<T> Send for SharedQueueMut<T> {}
unsafe impl<T> Sync for SharedQueueMut<T> {}

impl<T> Clone for SharedSlice<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
        }
    }
}

impl<T> Clone for SharedSliceMut<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
        }
    }
}

#[allow(dead_code)]
impl<T> AbstractedProceduralMemory<T> {
    pub fn from_file_name(file_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .read(true)
            .truncate(false)
            .write(false)
            .open(file_name)?;
        let (slice, mmap) = SharedSlice::<T>::from_file(&file)?;
        let mmap = Arc::new(mmap);

        Ok(AbstractedProceduralMemory {
            slice,
            mmap,
            _vec: Vec::new(),
            mmapped: true,
        })
    }
    #[inline(always)]
    pub(crate) fn shared_slice(&self) -> SharedSlice<T> {
        self.slice.clone()
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> &T {
        self.slice.get(idx)
    }

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.slice.ptr
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.slice.ptr, self.slice.len) }
    }

    #[allow(unused_mut)]
    #[inline(always)]
    pub(crate) fn into_bodex_slice(mut self) -> Box<[T]> {
        let this = ManuallyDrop::new(self);
        let raw: *mut [T] =
            std::ptr::slice_from_raw_parts_mut(this.slice.ptr as *mut T, this.slice.len);
        // SAFETY: `raw` points to a uniquely-owned heap allocation of exactly `len` Ts,
        unsafe { Box::from_raw(raw) }
    }

    #[inline(always)]
    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        self.slice.slice(start, end)
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.shared_slice().len()
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.shared_slice().len() == 0
    }
}

#[allow(dead_code)]
impl<T> AbstractedProceduralMemoryMut<T> {
    /// This is an immutable array! Used
    pub(crate) fn from_file_name(file_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .read(true)
            .truncate(false)
            .write(true)
            .open(file_name)?;
        let (slice, mmap) = SharedSliceMut::<T>::from_file(&file)?;
        let mmap = Arc::new(mmap);

        Ok(AbstractedProceduralMemoryMut {
            slice,
            mmap: Arc::try_unwrap(mmap).map_err(|_| -> Box<dyn std::error::Error> {
                "error couldn't dereference Mmap shared pointer".into()
            })?,
            _vec: Vec::new(),
            mmapped: true,
        })
    }

    /// Guarantees file is long enough for the mutable array, but doesn't tuncate it if it's
    /// longer than what is asked for.
    pub(crate) fn from_file(file: &File, len: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let file_len = file.metadata()?.len();
        let mmap_len = (len * size_of::<T>()).max(1) as u64;
        if file_len < mmap_len {
            file.set_len(mmap_len)?;
        }
        let (slice, mmap) = SharedSliceMut::<T>::from_file(file)?;
        mmap.flush()?;

        Ok(AbstractedProceduralMemoryMut {
            slice,
            mmap,
            _vec: Vec::new(),
            mmapped: true,
        })
    }

    #[inline(always)]
    pub(crate) fn shared_slice(&self) -> SharedSliceMut<T> {
        self.slice.clone()
    }

    #[inline(always)]
    pub fn get(&self, idx: usize) -> &T {
        self.slice.get(idx)
    }

    #[inline(always)]
    pub fn get_mut(&mut self, idx: usize) -> &mut T {
        self.slice.get_mut(idx)
    }

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.slice.ptr
    }

    #[inline(always)]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.slice.ptr
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.slice.ptr, self.slice.len) }
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.slice.ptr, self.slice.len) }
    }

    #[allow(unused_mut)]
    #[inline(always)]
    pub(crate) fn into_bodex_slice(mut self) -> Box<[T]> {
        let this = ManuallyDrop::new(self);
        let raw: *mut [T] = std::ptr::slice_from_raw_parts_mut(this.slice.ptr, this.slice.len);
        // SAFETY: `raw` points to a uniquely-owned heap allocation of exactly `len` Ts,
        unsafe { Box::from_raw(raw) }
    }

    #[inline(always)]
    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        self.slice.slice(start, end)
    }

    #[inline(always)]
    pub(crate) fn mut_slice(&mut self, start: usize, end: usize) -> Option<&mut [T]> {
        self.slice.mut_slice(start, end)
    }

    #[inline(always)]
    pub(crate) fn write_slice(&mut self, idx: usize, slice: &[T]) -> Option<usize> {
        self.slice.write_slice(idx, slice)
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.shared_slice().len()
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.shared_slice().len() == 0
    }

    pub fn flush(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.mmapped {
            Ok(self.mmap.flush()?)
        } else {
            Ok(())
        }
    }

    pub(crate) fn flush_async(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.mmapped {
            Ok(self.mmap.flush_async()?)
        } else {
            Ok(())
        }
    }
}

#[allow(dead_code)]
impl<T> SharedSlice<T> {
    pub(crate) fn new(ptr: *const T, len: usize) -> Self {
        SharedSlice::<T> { ptr, len }
    }

    pub(crate) fn from_slice(slice: &[T]) -> Self {
        SharedSlice::<T> {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }

    #[allow(clippy::ptr_arg)]
    pub(crate) fn from_vec(vec: &Vec<T>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::new(vec.as_ptr(), vec.len()))
    }

    pub(crate) fn from_file(file: &File) -> Result<(Self, Mmap), Box<dyn std::error::Error>> {
        let file_len = file.metadata()?.len() as usize;
        // Ensure the memory is properly aligned (a file shorter than an element holds none: empty
        // arrays are stored as 1 byte, as memory maps can't be empty)
        if size_of::<T>() != 0 && file_len % size_of::<T>() != 0 && file_len >= size_of::<T>() {
            return Err("file length is not a multiple of u64".into());
        } else if size_of::<T>() == 0 && file_len != 1 {
            // Memory mapped files may not be length zero ;)
            return Err("file length for o length types should be 1".into());
        }

        let mmap = unsafe { MmapOptions::new().len(file_len).map(file)? };
        let len = if size_of::<T>() == 0 {
            1
        } else {
            file_len / size_of::<T>()
        };

        Ok((Self::new(mmap.as_ptr() as *const T, len), mmap))
    }

    #[inline(always)]
    pub(self) fn ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn get(&self, idx: usize) -> &T {
        assert!(idx < self.len);
        unsafe { &*self.ptr.add(idx) }
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    // FIXME: if `mut` is removed does unique ownership get comprimized? Is it already compromized?
    #[allow(unused_mut)]
    #[inline(always)]
    pub(crate) fn into_bodex_slice(mut self) -> Box<[T]> {
        let this = ManuallyDrop::new(self);
        let raw: *mut [T] = std::ptr::slice_from_raw_parts_mut(this.ptr as *mut T, this.len);
        // SAFETY: `raw` points to a uniquely-owned heap allocation of exactly `len` Ts,
        unsafe { Box::from_raw(raw) }
    }

    #[inline(always)]
    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        if start > self.len {
            return None;
        }
        let end = if end > self.len { self.len } else { end };
        unsafe { Some(std::slice::from_raw_parts(self.ptr.add(start), end - start)) }
    }

    pub fn abstract_mem(
        mfn: &str,
        vec: Vec<T>,
        len: usize,
        mmapped: bool,
    ) -> Result<AbstractedProceduralMemory<T>, Box<dyn std::error::Error>> {
        let file = OpenOptions::new().read(true).open(mfn)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let mmap = Arc::new(mmap);

        // Ensure the memory is properly aligned
        if mmapped && (size_of::<T>() == 0 && mmap.len() != 1 || mmap.len() < size_of::<T>() * len)
        {
            return Err("file length is too small for the amount of elements requested".into());
        } else if !mmapped && vec.len() < len {
            return Err("vec length is too small for the amount of elements requested".into());
        }

        let slice: SharedSlice<T> = {
            if mmapped {
                SharedSlice::<T>::new(mmap.as_ptr() as *const T, len)
            } else {
                SharedSlice::<T>::new(vec.as_ptr(), len)
            }
        };

        Ok(AbstractedProceduralMemory {
            slice,
            _vec: vec,
            mmap,
            mmapped,
        })
    }
}

#[allow(dead_code)]
impl<T> SharedSliceMut<T> {
    pub(crate) fn new(ptr: *mut T, len: usize) -> Self {
        SharedSliceMut::<T> { ptr, len }
    }

    pub(crate) fn from_slice(slice: &mut [T]) -> Self {
        SharedSliceMut::<T> {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    pub(crate) fn from_shared_slice(slice: SharedSliceMut<T>) -> Self {
        SharedSliceMut::<T> {
            ptr: slice.ptr,
            len: slice.len,
        }
    }

    pub(crate) fn from_vec(vec: &mut Vec<T>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::new(vec.as_mut_ptr(), vec.len()))
    }

    /// Requires file length to be a multiple of size_of::<T>() (if T is not a zero-sized type,
    /// inwhich case file length has to be 1).
    pub(crate) fn from_file(file: &File) -> Result<(Self, MmapMut), Box<dyn std::error::Error>> {
        let mmap = unsafe { MmapOptions::new().map_mut(file)? };
        let mmap_len = mmap.len();

        // Ensure the memory is properly aligned (a file shorter than an element holds none: empty
        // arrays are stored as 1 byte, as memory maps can't be empty)
        if size_of::<T>() != 0 && mmap_len % size_of::<T>() != 0 && mmap_len >= size_of::<T>() {
            return Err("file length is not a multiple of u64".into());
        } else if size_of::<T>() == 0 && mmap_len != 1 {
            return Err("file length for 0 length types should be 1".into());
        }

        let len = if size_of::<T>() == 0 {
            1
        } else {
            mmap_len / size_of::<T>()
        };

        Ok((Self::new(mmap.as_ptr() as *mut T, len), mmap))
    }

    #[inline(always)]
    pub(crate) unsafe fn cast<U>(&self) -> Option<SharedSliceMut<U>> {
        if size_of::<T>() != size_of::<U>() {
            return None;
        }
        Some(SharedSliceMut {
            ptr: self.ptr as *mut U,
            len: self.len,
        })
    }

    #[inline(always)]
    pub(self) fn ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn get_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.len, "index {} >= len {}", idx, self.len);
        unsafe { &mut *self.ptr.add(idx) }
    }

    #[inline(always)]
    pub(crate) fn get(&self, idx: usize) -> &T {
        assert!(idx < self.len, "index {} >= len {}", idx, self.len);
        unsafe { &*self.ptr.add(idx) }
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    #[allow(unused_mut)]
    #[inline(always)]
    pub(crate) fn into_bodex_slice(mut self) -> Box<[T]> {
        let this = ManuallyDrop::new(self);
        let raw: *mut [T] = std::ptr::slice_from_raw_parts_mut(this.ptr, this.len);
        // SAFETY: `raw` points to a uniquely-owned heap allocation of exactly `len` Ts,
        unsafe { Box::from_raw(raw) }
    }

    #[inline(always)]
    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        assert!(start <= end && end <= self.len);
        unsafe { Some(std::slice::from_raw_parts(self.ptr.add(start), end - start)) }
    }

    #[inline(always)]
    pub(crate) fn mut_slice(&mut self, start: usize, end: usize) -> Option<&mut [T]> {
        assert!(start <= end && end <= self.len);
        unsafe {
            Some(std::slice::from_raw_parts_mut(
                self.ptr.add(start),
                end - start,
            ))
        }
    }

    // SAFETY: Only if 0 is a valid bit-pattern for T (true for integers/floats).
    fn zero_out(&mut self, start: usize, end: usize) {
        assert!(start <= end && end <= self.len);
        unsafe {
            std::ptr::write_bytes(self.ptr.add(start), 0, end - start);
        }
    }

    // SAFETY: Only if 0 is a valid bit-pattern for T (true for integers/floats).
    pub(crate) fn zero_out_bytes(&mut self) -> Self {
        unsafe {
            std::ptr::write_bytes(self.ptr, 0, self.len);
        }
        self.clone()
    }

    /// SAFETY: Only if 0 is a valid bit-pattern for T (true for integers/floats).
    pub(crate) fn par_zero_out_bytes(&mut self) -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let chunks = cores.min(8); // prevent oversharding
        let len = self.len();
        let chunk = len.div_ceil(chunks);

        std::thread::scope(|s| {
            for i in 0..chunks {
                let mut m = self.clone();
                let start = std::cmp::min(i * chunk, len);
                let end = std::cmp::min(start + chunk, len);
                s.spawn(move || {
                    m.zero_out(start, end);
                });
            }
        });
        self.clone()
    }

    #[inline(always)]
    pub(crate) fn write_shared_slice(
        &mut self,
        slice: SharedSlice<T>,
        idx: usize,
        from: usize,
        len: usize,
    ) -> Option<usize> {
        assert!(idx + len <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(slice.ptr().add(from), self.ptr.add(idx), len);
        };
        Some(idx + slice.len())
    }

    #[inline(always)]
    pub(crate) fn write_shared_slice_mut(
        &mut self,
        slice: SharedSliceMut<T>,
        idx: usize,
        from: usize,
        len: usize,
    ) -> Option<usize> {
        assert!(idx + len <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(slice.ptr().add(from), self.ptr.add(idx), len);
        };
        Some(idx + slice.len())
    }

    #[inline(always)]
    pub(crate) fn write_slice(&mut self, idx: usize, slice: &[T]) -> Option<usize> {
        // if idx == len but slice.len() == 0, the write should be valid
        assert!(idx + slice.len() <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), self.ptr.add(idx), slice.len());
        };
        Some(idx + slice.len())
    }

    /// Truncates file!
    pub fn abst_mem_mut(
        mfn: &str,
        len: usize,
        mmapped: bool,
    ) -> Result<AbstractedProceduralMemoryMut<T>, Box<dyn std::error::Error>> {
        // a new file, rather than the old one truncated: whoever still maps the old one (another
        // instance of the graph, an earlier result) keeps it, intact
        let file = create_new_file(mfn)?;
        let file_len = (if mmapped { len } else { 1 }
            * if size_of::<T>() == 0 {
                1
            } else {
                size_of::<T>()
            }) as u64;
        file.set_len(file_len)?;

        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        let mut vec: Vec<T> = Vec::new();

        if !mmapped {
            vec.reserve_exact(len);
        }

        let slice: SharedSliceMut<T> = {
            if mmapped {
                SharedSliceMut::<T>::new(mmap.as_mut_ptr() as *mut T, len)
            } else {
                SharedSliceMut::<T>::new(vec.as_mut_ptr(), len)
            }
        };

        Ok(AbstractedProceduralMemoryMut {
            slice,
            _vec: vec,
            mmap,
            mmapped,
        })
    }
}

#[allow(dead_code)]
impl<T: Debug + Copy + Clone + Eq> SharedQueueMut<T> {
    pub(crate) fn from_shared_slice(slice: SharedSliceMut<T>) -> Self {
        SharedQueueMut::<T> {
            ptr: slice.ptr,
            max: slice.len,
            read: Arc::new(AtomicUsize::new(0)),
            write: Arc::new(AtomicUsize::new(0)),
            len: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.len.load(Ordering::Relaxed) == 0
    }

    pub(crate) fn push_slice(&mut self, slice: &[T]) -> Option<usize> {
        let write_idx = self.write.fetch_add(slice.len(), Ordering::SeqCst);
        // `<=`: a slice filling the queue exactly fits
        if write_idx + slice.len() <= self.max {
            unsafe {
                std::ptr::copy_nonoverlapping(slice.as_ptr(), self.ptr.add(write_idx), slice.len());
            };
            self.len.fetch_add(slice.len(), Ordering::SeqCst);
            Some(write_idx)
        } else {
            println!(
                "Failed to push write_idx {} slice len {}",
                write_idx,
                slice.len()
            );
            self.write.fetch_sub(slice.len(), Ordering::SeqCst);
            None
        }
    }

    pub(crate) fn push(&mut self, el: T) -> Option<usize> {
        let write_idx = self.write.fetch_add(1, Ordering::SeqCst);
        if write_idx < self.max {
            unsafe { *self.ptr.add(write_idx) = el };
            self.len.fetch_add(1, Ordering::SeqCst);
            Some(write_idx)
        } else {
            // println!("Failed to push");
            None
        }
    }

    /// Never decrement length! read from beginning to avoid contention problems with concurrent push() / pop().
    pub(crate) fn pop(&mut self) -> Option<T> {
        let idx = self
            .read
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                if x < self.len.load(Ordering::SeqCst) {
                    Some(x + 1)
                } else {
                    None
                }
            })
            .ok();
        if let Some(offset) = idx {
            Some(unsafe { *self.ptr.add(offset) })
        } else {
            None
        }
    }

    /// Pops up to `max` elements at once (fewer contended updates than [`pop`](Self::pop)ping
    /// them one by one), or `None` if the queue is empty.
    pub(crate) fn pop_chunk(&mut self, max: usize) -> Option<&[T]> {
        let start = self
            .read
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                let len = self.len.load(Ordering::SeqCst);
                if x < len {
                    Some(std::cmp::min(x + max, len))
                } else {
                    None
                }
            })
            .ok()?;
        let end = std::cmp::min(start + max, self.len.load(Ordering::SeqCst));
        // SAFETY: `start..end` was claimed (exclusively) by the update above, and was pushed
        Some(unsafe { std::slice::from_raw_parts(self.ptr.add(start), end - start) })
    }

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len()) }
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len()) }
    }

    #[allow(unused_mut)]
    #[inline(always)]
    /// Returns a boxed slice over the entirity of the allocated memory[^1].
    ///
    /// [^1]: naturally, if any uninitialized memory cells remain at the time of this method's calling,
    /// behavioiur upon their reading is undefined.
    pub(crate) fn into_bodex_slice(mut self) -> Box<[T]> {
        let this = ManuallyDrop::new(self);
        let raw: *mut [T] = std::ptr::slice_from_raw_parts_mut(this.ptr, this.max);
        // SAFETY: `raw` points to a uniquely-owned heap allocation of exactly `len` Ts,
        unsafe { Box::from_raw(raw) }
    }

    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<SharedSliceMut<T>> {
        let start = if start < self.read.load(Ordering::Relaxed) {
            self.read.load(Ordering::Relaxed)
        } else {
            start
        };
        let end = if end > self.len.load(Ordering::Relaxed) {
            self.len.load(Ordering::Relaxed)
        } else {
            if end < start {
                return None;
            }
            end
        };
        unsafe { Some(SharedSliceMut::new(self.ptr.add(start), end - start)) }
    }

    pub(crate) unsafe fn raw_slice(&self) -> SharedSliceMut<T> {
        SharedSliceMut::new(self.ptr, self.max)
    }

    pub(crate) fn clear(self) -> Self {
        self.read.store(0, Ordering::SeqCst);
        self.write.store(0, Ordering::SeqCst);
        self.len.store(0, Ordering::SeqCst);
        self
    }
}

/// A scratch array in a memmapped file of its own, removed once the array is dropped: an
/// algorithm's working memory, kept out of the heap however large the graph. Zeroed when
/// created.
///
/// Its elements are never dropped: meant for plain data.
#[derive(Debug)]
pub(crate) struct ScratchMem<T> {
    mem: ManuallyDrop<AbstractedProceduralMemoryMut<T>>,
    file_name: String,
}

#[allow(dead_code)]
impl<T> ScratchMem<T> {
    /// A zeroed array of `len` elements, in a new file at `file_name` (in place of any there).
    pub(crate) fn new(file_name: String, len: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mem = SharedSliceMut::<T>::abst_mem_mut(&file_name, len, true)?;
        Ok(Self {
            mem: ManuallyDrop::new(mem),
            file_name,
        })
    }

    /// An array of `len` elements, all `value`.
    pub(crate) fn filled(
        file_name: String,
        len: usize,
        value: T,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        T: Copy,
    {
        let mut mem = Self::new(file_name, len)?;
        mem.fill(value);
        Ok(mem)
    }

    #[inline(always)]
    pub(crate) fn shared_slice(&self) -> SharedSliceMut<T> {
        self.mem.shared_slice()
    }
}

impl<T> std::ops::Deref for ScratchMem<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.mem.as_slice()
    }
}

impl<T> std::ops::DerefMut for ScratchMem<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        self.mem.as_mut_slice()
    }
}

impl<T> Drop for ScratchMem<T> {
    fn drop(&mut self) {
        // unmapped first: some platforms won't remove a mapped file
        unsafe { ManuallyDrop::drop(&mut self.mem) };
        let _ = remove_if_exists(&self.file_name);
    }
}

/// A stack of at most a given number of elements, in a memmapped scratch file (see
/// [`ScratchMem`]): for depth first searches and the like, whose stacks may grow as large as the
/// graph.
///
/// Unlike [`ScratchMem`]'s, its elements are owned (and dropped) as a [`Vec`]'s are.
#[derive(Debug)]
pub(crate) struct ScratchStack<T> {
    mem: ScratchMem<std::mem::MaybeUninit<T>>,
    len: usize,
}

#[allow(dead_code)]
impl<T> ScratchStack<T> {
    /// An empty stack of room for `capacity` elements, in a new file at `file_name`.
    pub(crate) fn new(
        file_name: String,
        capacity: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            mem: ScratchMem::new(file_name, capacity)?,
            len: 0,
        })
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub(crate) fn capacity(&self) -> usize {
        self.mem.len()
    }

    /// Pushes `value` on the stack.
    ///
    /// # Panics
    ///
    /// If the stack is full.
    #[inline(always)]
    pub(crate) fn push(&mut self, value: T) {
        assert!(
            self.len < self.capacity(),
            "error scratch stack of {} elements is full",
            self.capacity()
        );
        self.mem[self.len].write(value);
        self.len += 1;
    }

    #[inline(always)]
    pub(crate) fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // initialized (pushed), and no longer the stack's
        Some(unsafe { self.mem[self.len].assume_init_read() })
    }

    #[inline(always)]
    pub(crate) fn last(&self) -> Option<&T> {
        self.as_slice().last()
    }

    #[inline(always)]
    pub(crate) fn last_mut(&mut self) -> Option<&mut T> {
        self.as_mut_slice().last_mut()
    }

    /// Removes (and drops) every element.
    pub(crate) fn clear(&mut self) {
        let elements: *mut [T] = self.as_mut_slice();
        self.len = 0;
        // initialized, and no longer the stack's
        unsafe { std::ptr::drop_in_place(elements) };
    }

    /// The stack's elements, bottom first.
    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        // the first `len` are initialized
        unsafe { std::slice::from_raw_parts(self.mem.as_ptr() as *const T, self.len) }
    }

    /// The stack's elements, bottom first.
    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        // the first `len` are initialized
        unsafe { std::slice::from_raw_parts_mut(self.mem.as_mut_ptr() as *mut T, self.len) }
    }
}

impl<T> Drop for ScratchStack<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::{Mmap, MmapMut};
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::mem::size_of;
    use std::path::{Path, PathBuf};

    #[test]
    fn scratch_mem_is_zeroed_and_removed_when_dropped() {
        let dir = std::env::temp_dir().join(format!("scratch_mem_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mem.tmp").to_string_lossy().into_owned();
        {
            let mut mem = ScratchMem::<u64>::new(path.clone(), 1000).unwrap();
            assert!(mem.iter().all(|&x| x == 0));
            mem[999] = 7;
            assert_eq!(mem[999], 7);
            assert!(Path::new(&path).exists());
            let filled =
                ScratchMem::<u32>::filled(dir.join("f.tmp").to_string_lossy().into_owned(), 3, 9)
                    .unwrap();
            assert_eq!(&filled[..], &[9, 9, 9]);
        }
        assert!(!Path::new(&path).exists());
        assert!(!dir.join("f.tmp").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn scratch_stack_behaves_as_a_vec() {
        use std::rc::Rc;
        let dir = std::env::temp_dir().join(format!("scratch_stack_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stack.tmp").to_string_lossy().into_owned();
        let mut s = ScratchStack::<(u32, u64)>::new(path.clone(), 3).unwrap();
        assert!(s.is_empty() && s.pop().is_none() && s.last().is_none());
        s.push((1, 10));
        s.push((2, 20));
        s.last_mut().unwrap().1 += 1;
        assert_eq!(s.as_slice(), &[(1, 10), (2, 21)]);
        assert_eq!(s.pop(), Some((2, 21)));
        s.push((3, 30));
        s.push((4, 40));
        assert_eq!(s.len(), 3);
        s.as_mut_slice().reverse();
        assert_eq!(s.as_slice(), &[(4, 40), (3, 30), (1, 10)]);
        s.clear();
        assert!(s.is_empty());
        drop(s);
        assert!(!Path::new(&path).exists());

        // owned elements are dropped exactly once: popped, cleared or left on the stack
        let rc = Rc::new(());
        let mut s = ScratchStack::<Rc<()>>::new(path.clone(), 4).unwrap();
        for _ in 0..4 {
            s.push(rc.clone());
        }
        drop(s.pop());
        assert_eq!(Rc::strong_count(&rc), 4);
        drop(s);
        assert_eq!(Rc::strong_count(&rc), 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    #[should_panic(expected = "full")]
    fn scratch_stack_overflow_panics() {
        let path = std::env::temp_dir()
            .join(format!("scratch_stack_full_{}.tmp", std::process::id()))
            .to_string_lossy()
            .into_owned();
        let mut s = ScratchStack::<u8>::new(path, 1).unwrap();
        s.push(1);
        s.push(2);
    }

    // ---------------- helpers ----------------

    /// Create a unique temp file path in the OS temp dir.
    fn temp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        // add pid + time for uniqueness without extra crates
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("sharedslice_{pid}_{nanos}_{name}"));
        p
    }

    /// Create a file with the given raw bytes, return the opened File (read-only).
    fn create_file_with_bytes(path: &Path, bytes: &[u8]) -> File {
        // create/truncate and write
        {
            let mut f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(path)
                .expect("create temp file");
            f.write_all(bytes).expect("write bytes");
            f.sync_all().ok();
        }
        // reopen read-only to match function signature expectations
        OpenOptions::new()
            .read(true)
            .open(path)
            .expect("reopen readonly")
    }

    fn create_rw_file_with_bytes(path: &Path, bytes: &[u8]) -> File {
        // write and keep a read+write handle (for map_mut)
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)
            .expect("create temp file");
        f.write_all(bytes).expect("write bytes");
        f.sync_all().ok();
        f
    }

    fn write_bytes(path: &Path, bytes: &[u8]) {
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(path)
            .expect("create tmp");
        f.write_all(bytes).expect("write");
        f.sync_all().ok();
    }

    fn read_all(path: &Path) -> Vec<u8> {
        let mut v = Vec::new();
        let mut f = OpenOptions::new().read(true).open(path).expect("open ro");
        f.read_to_end(&mut v).expect("read");
        v
    }

    // Compile-time check: SharedSlice<T> is Send+Sync for any T (as declared).
    fn assert_send_sync<T: Send + Sync>() {}
    #[test]
    fn sharedslice_is_send_sync() {
        assert_send_sync::<SharedSlice<u64>>();
        assert_send_sync::<SharedSlice<()>>();
    }

    // ---------------- from_slice / basics ----------------

    #[test]
    fn from_slice_and_access() {
        let a: [u32; 3] = [1, 2, 3];
        let ss = SharedSlice::from_slice(&a);
        assert_eq!(ss.len(), 3);
        assert!(!ss.is_empty());
        assert_eq!(*ss.get(0), 1);
        assert_eq!(*ss.get(1), 2);
        assert_eq!(*ss.get(2), 3);
        assert_eq!(ss.as_slice(), &a);
    }

    // ---------------- from_vec (pointer/len stay valid while vec lives) ----------------

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn from_vec_while_vec_alive() {
        let v = vec![10usize, 20, 30, 40];
        let ss = SharedSlice::from_vec(&v).unwrap();
        // use without dropping v
        assert_eq!(ss.len(), v.len());
        assert_eq!(ss.as_slice(), &v[..]);
        assert_eq!(*ss.get(3), 40);
        // ensure clone reuses same view
        let ss2 = ss.clone();
        assert_eq!(ss2.as_slice(), ss.as_slice());
    }

    // ---------------- slice() bounds and truncation ----------------

    #[test]
    fn slice_within_and_truncated_end() {
        let a = [5u8, 6, 7, 8, 9];
        let ss = SharedSlice::from_slice(&a);
        // within bounds
        let s = ss.slice(1, 4).expect("slice");
        assert_eq!(s, &[6, 7, 8]);

        // end beyond len should truncate to len
        let s2 = ss.slice(3, 99).expect("slice");
        assert_eq!(s2, &[8, 9]);

        // start > len -> None
        assert!(ss.slice(6, 7).is_none());
        // NOTE: start > end is undefined in this API; we do not test it to avoid UB.
    }

    // ---------------- from_file<u64>: ok path ----------------

    #[test]
    fn from_file_u64_ok() {
        // prepare 3 u64s
        let vals: [u64; 3] = [10, 20, 30];
        let bytes = unsafe {
            std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * size_of::<u64>())
        };
        let p = temp_path("u64_ok.bin");
        let f = create_file_with_bytes(&p, bytes);

        let (ss, _mmap): (SharedSlice<u64>, Mmap) = SharedSlice::from_file(&f).expect("map file");
        assert_eq!(ss.len(), 3);
        assert_eq!(*ss.get(0), 10);
        assert_eq!(*ss.get(1), 20);
        assert_eq!(*ss.get(2), 30);

        let _ = fs::remove_file(p);
    }

    // ---------------- from_file<u64>: bad length (not multiple of size) ----------------

    #[test]
    fn from_file_u64_bad_len_errors() {
        // file length 10 (not multiple of 8)
        let p = temp_path("u64_badlen.bin");
        {
            let mut f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(&p)
                .unwrap();
            f.set_len(10).unwrap();
            f.seek(SeekFrom::Start(10)).unwrap();
            f.flush().ok();
        }
        let f = OpenOptions::new().read(true).open(&p).unwrap();
        let err = SharedSlice::<u64>::from_file(&f).unwrap_err();
        let s = err.to_string();
        assert!(
            s.contains("file length is not a multiple of u64"),
            "unexpected error: {s}"
        );
        let _ = fs::remove_file(p);
    }

    // ---------------- from_file<()>: zero-sized type (zst) path ----------------

    #[test]
    fn from_file_zst_ok_when_len_is_one() {
        // for ZST, function requires file length == 1
        let p = temp_path("zst_len1.bin");
        {
            let mut f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(&p)
                .unwrap();
            f.set_len(1).unwrap();
            f.write_all(&[0u8]).unwrap();
            f.flush().ok();
        }
        let f = OpenOptions::new().read(true).open(&p).unwrap();
        let (ss, _mmap): (SharedSlice<()>, Mmap) = SharedSlice::from_file(&f).unwrap();
        assert_eq!(ss.len(), 1);
        // Can take a reference to the unit value
        let _unit: &() = ss.get(0);
        let _ = fs::remove_file(p);
    }

    #[test]
    fn from_file_zst_wrong_len_errors() {
        // for ZST, length must be exactly 1
        let p = temp_path("zst_len0.bin");
        {
            let mut f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(&p)
                .unwrap();
            f.set_len(0).unwrap();
            f.flush().ok();
        }
        let f = OpenOptions::new().read(true).open(&p).unwrap();
        let err = SharedSlice::<()>::from_file(&f).unwrap_err();
        assert!(
            err.to_string()
                .contains("file length for o length types should be 1"),
            "unexpected error: {}",
            err
        );
        let _ = fs::remove_file(p);
    }

    // ---------------- as_slice matches get loop ----------------

    #[test]
    fn as_slice_matches_get() {
        let v = [100i32, 200, 300, 400, 500];
        let ss = SharedSlice::from_slice(&v[..]);
        let s = ss.as_slice();
        (0..ss.len()).for_each(|i| {
            assert_eq!(s[i], *ss.get(i));
        });
    }

    #[test]
    fn sharedslice_mut_is_send_sync() {
        assert_send_sync::<SharedSliceMut<u64>>();
        assert_send_sync::<SharedSliceMut<()>>();
    }

    // ---------- from_slice / basic access ----------
    #[test]
    fn from_slice_mut_and_access() {
        let mut a = [1u32, 2, 3, 4];
        let mut ss = SharedSliceMut::from_slice(&mut a);
        assert_eq!(ss.len(), 4);
        assert!(!ss.is_empty());
        assert_eq!(*ss.get(0), 1);
        *ss.get_mut(1) = 20;
        assert_eq!(a[1], 20);

        // as_mut_slice roundtrip
        ss.as_mut_slice().swap(2, 3);
        assert_eq!(a, [1, 20, 4, 3]);
    }

    // ---------- from_vec (reflect changes) ----------
    #[test]
    #[allow(clippy::clone_on_copy)]
    fn from_vec_reflects_changes() {
        let mut v = vec![10usize, 20, 30];
        let mut ss = SharedSliceMut::from_vec(&mut v).unwrap();
        ss.as_mut_slice()[1] = 200;
        assert_eq!(v, vec![10, 200, 30]);

        // clone keeps same view
        let ss2 = ss.clone();
        assert_eq!(ss2.as_slice(), ss.as_slice());
    }

    // ---------- slicing (immut/mut) ----------
    #[test]
    fn slice_and_mut_slice_ranges() {
        let mut a = [5u8, 6, 7, 8, 9];
        let mut ss = SharedSliceMut::from_slice(&mut a);
        let s = ss.slice(1, 4).unwrap();
        assert_eq!(s, &[6, 7, 8]);

        {
            let m = ss.mut_slice(2, 5).unwrap();
            m.copy_from_slice(&[70, 80, 90]);
        }
        assert_eq!(a, [5, 6, 70, 80, 90]);
    }

    #[test]
    fn from_file_u64_ok_and_write() {
        let vals: [u64; 3] = [100, 200, 300];
        let bytes = unsafe {
            std::slice::from_raw_parts(
                vals.as_ptr() as *const u8,
                vals.len() * std::mem::size_of::<u64>(),
            )
        };
        let p = temp_path("u64_ok_mut.bin");
        let f = create_rw_file_with_bytes(&p, bytes); // <-- was read-only

        let (mut ss, mmap): (SharedSliceMut<u64>, MmapMut) =
            SharedSliceMut::from_file(&f).expect("map");
        assert_eq!(ss.len(), 3);
        assert_eq!(*ss.get(1), 200);
        *ss.get_mut(1) = 222;
        mmap.flush().ok();
        let _ = fs::remove_file(p);
    }

    #[test]
    fn from_file_u64_bad_len_errors_2() {
        // length 10 (not multiple of 8)
        let p = temp_path("u64_badlen_mut.bin");
        let f = {
            let f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(&p)
                .unwrap();
            f.set_len(10).unwrap();
            f // keep RW handle for map_mut
        };

        let err = SharedSliceMut::<u64>::from_file(&f).unwrap_err();
        assert!(
            err.to_string()
                .contains("file length is not a multiple of u64"),
            "unexpected err: {err}"
        );
        let _ = fs::remove_file(p);
    }

    #[test]
    fn from_file_zst_len1_ok_and_wrong_len_errors() {
        // ok: len==1
        let p1 = temp_path("zst_len1_mut.bin");
        let f1 = {
            let mut f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(&p1)
                .unwrap();
            f.set_len(1).unwrap();
            f.write_all(&[0]).unwrap();
            f // keep RW handle for map_mut
        };
        let (ss, _mmap) = SharedSliceMut::<()>::from_file(&f1).unwrap();
        assert_eq!(ss.len(), 1);
        let _ = fs::remove_file(&p1);

        // bad: len != 1
        let p0 = temp_path("zst_len0_mut.bin");
        let f0 = {
            let f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(&p0)
                .unwrap();
            f.set_len(0).unwrap();
            f
        };
        let err = SharedSliceMut::<()>::from_file(&f0).unwrap_err();
        assert!(
            err.to_string()
                .contains("file length for 0 length types should be 1"),
            "unexpected err: {err}"
        );
        let _ = fs::remove_file(p0);
    }

    // ---------- cast<U> ----------
    #[test]
    fn cast_same_size_ok_and_diff_size_none() {
        let mut a = [1u32, 2, 3, 4];
        let ss = SharedSliceMut::from_slice(&mut a);

        // same size
        let b: Option<SharedSliceMut<u32>> = unsafe { ss.cast::<u32>() };
        assert!(b.is_some());
        assert_eq!(b.unwrap().len(), ss.len());

        // different size
        let c: Option<SharedSliceMut<u64>> = unsafe { ss.cast::<u64>() };
        assert!(c.is_none());
    }

    // ---------- write_slice ----------
    #[test]
    fn write_slice_copies_content() {
        let mut dst = [0i32; 6];
        let mut ss = SharedSliceMut::from_slice(&mut dst);
        let next = ss.write_slice(1, &[9, 8, 7]).unwrap();
        assert_eq!(next, 4); // 1 + 3
        assert_eq!(dst, [0, 9, 8, 7, 0, 0]);
    }

    // ---------- write_shared_slice / write_shared_slice_mut ----------
    #[test]
    fn write_shared_slice_variants_copy_content() {
        // prepare dst
        let mut dst = [0usize; 6];
        let mut dss = SharedSliceMut::from_slice(&mut dst);

        // immutable source
        let src = [10usize, 11, 12, 13];
        let ssrc = SharedSlice::from_slice(&src);

        // copy 4 elems from src[0..4] into dst starting at 2
        let r1 = dss.write_shared_slice(ssrc, 2, 0, 4).unwrap();
        assert_eq!(dst, [0, 0, 10, 11, 12, 13]);
        // return value is idx + slice.len() (quirk of current impl)
        assert_eq!(r1, 2 + ssrc.len());

        // mutable source
        let mut srcm: [usize; 5] = [100, 200, 255, 300, 400];
        let ssrcm = SharedSliceMut::from_slice(&mut srcm);
        let mut dss2 = SharedSliceMut::from_slice(&mut dst);
        let r2 = dss2.write_shared_slice_mut(ssrcm, 0, 1, 3).unwrap();
        assert_eq!(dst[0..3], [200, 255, 300]);
        assert_eq!(r2, ssrcm.len()); // same quirk
    }

    // ---------- out-of-bounds panics ----------
    #[test]
    #[should_panic(expected = "index")]
    fn get_panics_oob() {
        let mut a = [1u8, 2, 3];
        let ss = SharedSliceMut::from_slice(&mut a);
        let _ = ss.get(5);
    }

    #[test]
    #[should_panic(expected = "index")]
    fn get_mut_panics_oob() {
        let mut a = [1u8, 2, 3];
        let mut ss = SharedSliceMut::from_slice(&mut a);
        let _ = ss.get_mut(5);
    }

    // ---------- pointer getters ----------
    #[test]
    fn ptrs_are_consistent() {
        let mut a = [1i64, 2, 3];
        let mut ss = SharedSliceMut::from_slice(&mut a);
        assert_eq!(ss.as_ptr(), a.as_ptr());
        assert_eq!(ss.as_mut_ptr(), a.as_mut_ptr());
    }
    /// Build a queue with capacity `cap` backed by a Vec<u64> of zeros.
    /// Returns (queue, backing_buffer) — keep the Vec alive!
    fn make_queue_u64(cap: usize) -> (SharedQueueMut<u64>, Vec<u64>) {
        let mut buf = vec![0u64; cap];
        let ss = SharedSliceMut::from_slice(&mut buf[..]);
        let q = SharedQueueMut::from_shared_slice(ss);
        (q, buf)
    }

    // --- basic push/pop FIFO --------------------------------------------------

    #[test]
    fn queue_push_pop_fifo() {
        let (mut q, mut _buf) = make_queue_u64(8);

        assert_eq!(q.len(), 0);
        assert!(q.is_empty());

        // push three
        assert_eq!(q.push(10), Some(0));
        assert_eq!(q.push(20), Some(1));
        assert_eq!(q.push(30), Some(2));
        assert_eq!(q.len(), 3);

        // pop in FIFO order
        assert_eq!(q.pop(), Some(10));
        assert_eq!(q.pop(), Some(20));
        assert_eq!(q.pop(), Some(30));
        assert_eq!(q.pop(), None);
        assert!(q.len() >= 3, "len() counts pushes only in current impl");
        // We only assert pop order; len/read/write counters are internal.
    }

    // --- fill to capacity and overflow push ----------------------------------

    #[test]
    fn queue_push_until_full_then_fail() {
        let cap = 4;
        let (mut q, mut _buf) = make_queue_u64(cap);

        // push exactly `cap` items succeeds
        for i in 0..cap {
            assert_eq!(q.push(i as u64), Some(i));
        }
        // next push should fail
        assert_eq!(q.push(999), None);
    }

    // --- push_slice partial fit succeeds -------------------------------------

    #[test]
    fn push_slice_partial_fit_succeeds() {
        let (mut q, buf) = make_queue_u64(8);

        // Start with three singles
        assert_eq!(q.push(1), Some(0));
        assert_eq!(q.push(2), Some(1));
        assert_eq!(q.push(3), Some(2));

        // Now push a slice of length 4; 3 + 4 < 8 so it fits
        let start = q.push_slice(&[4, 5, 6, 7]).expect("slice should fit");
        assert_eq!(start, 3);

        // Verify backing buffer was written as expected
        assert_eq!(&buf[..7], &[1, 2, 3, 4, 5, 6, 7]);
    }

    // --- pop_chunk claims disjoint chunks covering every element once ---------

    #[test]
    fn pop_chunk_claims_each_element_once() {
        let (mut q, mut _buf) = make_queue_u64(10);
        q.push_slice(&[1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(q.pop_chunk(3), Some(&[1, 2, 3][..]));
        assert_eq!(q.pop(), Some(4));
        assert_eq!(q.pop_chunk(5), Some(&[5, 6, 7][..]));
        assert_eq!(q.pop_chunk(5), None);
        assert_eq!(q.pop(), None);
    }

    // --- push_slice exact fit (a full queue's worth) succeeds, one more doesn't ---

    #[test]
    fn push_slice_exact_fit_succeeds() {
        let (mut q, mut _buf) = make_queue_u64(5);
        assert_eq!(
            q.push_slice(&[1, 2, 3, 4, 5]),
            Some(0),
            "an exact fit should succeed"
        );
        assert_eq!(q.len(), 5);
        assert_eq!(
            q.push_slice(&[6]),
            None,
            "a full queue should reject more elements"
        );
        assert_eq!(q.len(), 5);
    }

    // --- slice() respects read pointer (clamps to unread region) --------------

    #[test]
    fn slice_clamps_to_unread_after_pops() {
        let (mut q, mut _buf) = make_queue_u64(10);

        // push six
        for i in 1..=6 {
            q.push(i);
        }
        // pop two
        assert_eq!(q.pop(), Some(1));
        assert_eq!(q.pop(), Some(2));

        // Ask for a slice that starts before `read`; it should clamp to unread
        let s = q.slice(0, 6).expect("slice");
        let s_ref = s.as_slice(); // SharedSliceMut<T> -> &[T]
        // Expect unread portion [3,4,5,6]
        assert_eq!(s_ref, &[3, 4, 5, 6]);
    }

    // --- clear() resets indices so pushes start from zero again ----------------

    #[test]
    fn clear_resets_and_allows_reuse() {
        let (mut q, buf) = make_queue_u64(4);

        assert_eq!(q.push(11), Some(0));
        assert_eq!(q.push(22), Some(1));
        let mut q = q.clear(); // consumes and returns reset queue

        // After clear, start from 0 again
        assert_eq!(q.push(99), Some(0));
        assert_eq!(buf[0], 99);
    }

    // --- pop from empty returns None -----------------------------------------

    #[test]
    fn pop_from_empty_is_none() {
        let (mut q, _buf) = make_queue_u64(3);
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn apm_from_file_name_u64_and_basic_ops() {
        // prepare 3 u64s
        let vals: [u64; 3] = [10, 20, 30];
        let bytes = unsafe {
            std::slice::from_raw_parts(vals.as_ptr() as *const u8, vals.len() * size_of::<u64>())
        };
        let p = temp_path("imm_u64.bin");
        write_bytes(&p, bytes);

        // immutable mapping from file name (RO is fine)
        let apm: AbstractedProceduralMemory<u64> =
            AbstractedProceduralMemory::from_file_name(p.to_str().unwrap()).expect("map ro");

        assert_eq!(apm.len(), 3);
        assert!(!apm.is_empty());
        assert_eq!(*apm.get(0), 10);
        assert_eq!(apm.as_slice(), &[10, 20, 30]);

        // slicing
        let s = apm.slice(1, 3).unwrap();
        assert_eq!(s, &[20, 30]);

        // shared_slice clone
        let ss = apm.shared_slice();
        assert_eq!(ss.as_slice(), &[10, 20, 30]);

        let _ = fs::remove_file(p);
    }

    // ---------------- AbstractedProceduralMemoryMut<T> (mutable) ----------------

    #[test]
    fn apm_mut_from_file_len_roundtrip_and_flush() {
        // Create empty file, open RW, and ask APMMut to size & map it
        let p = temp_path("mut_from_file_len.bin");
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&p)
            .expect("open rw");

        // map space for 4 u64s
        let mut apm: AbstractedProceduralMemoryMut<u64> =
            AbstractedProceduralMemoryMut::from_file(&file, 4).expect("map mut len");
        assert_eq!(apm.len(), 4);

        // write via as_mut_slice
        apm.as_mut_slice().copy_from_slice(&[100, 200, 300, 400]);
        assert_eq!(apm.as_slice(), &[100, 200, 300, 400]);

        // mutate a single cell
        *apm.get_mut(2) = 333;
        assert_eq!(apm.as_slice(), &[100, 200, 333, 400]);

        // slice / mut_slice
        let s = apm.slice(1, 4).unwrap();
        assert_eq!(s, &[200, 333, 400]);

        {
            let m = apm.mut_slice(0, 2).unwrap();
            m.copy_from_slice(&[7, 8]);
        }
        assert_eq!(apm.as_slice(), &[7, 8, 333, 400]);

        // flush to disk and verify bytes
        apm.flush().expect("flush");

        // read back and check the four u64s
        let raw = read_all(&p);
        assert_eq!(raw.len(), 4 * size_of::<u64>());
        let back: &[u64] = unsafe {
            std::slice::from_raw_parts(raw.as_ptr() as *const u64, raw.len() / size_of::<u64>())
        };
        assert_eq!(back, &[7, 8, 333, 400]);

        let _ = fs::remove_file(p);
    }

    // ---------------- tiny sanity for is_empty/len ----------------

    #[test]
    fn apm_len_and_empty_report() {
        let vals: [u8; 0] = [];
        let p = temp_path("imm_empty.bin");
        write_bytes(&p, &vals);

        let apm: AbstractedProceduralMemory<u8> =
            AbstractedProceduralMemory::from_file_name(p.to_str().unwrap()).expect("map ro");

        assert_eq!(apm.len(), 0);
        assert!(apm.is_empty());
        assert_eq!(apm.as_slice(), &[]);

        let _ = fs::remove_file(p);
    }

    #[test]
    fn apm_mut_from_file_name_does_not_zero_or_resize_and_persists_changes() {
        // Prepare a file with non-zero, distinctive content.
        let path = temp_path("mut_no_zero.bin");
        let original: Vec<u8> = (0..64).map(|i| (i as u8) ^ 0xA5).collect();
        write_bytes(&path, &original);
        let orig_len = fs::metadata(&path).unwrap().len();

        // Map mutably via from_file_name (must open with .write(true) internally).
        let mut apm = AbstractedProceduralMemoryMut::<u8>::from_file_name(path.to_str().unwrap())
            .expect("map_mut should work and not zero/resize");

        // 1) Contents are untouched (no zeroing).
        assert_eq!(apm.as_slice(), &original[..]);

        // 2) File size unchanged (no resize/truncate).
        assert_eq!(fs::metadata(&path).unwrap().len(), orig_len);

        // Mutate a single byte and flush.
        apm.as_mut_slice()[10] ^= 0xFF;
        apm.flush().expect("flush");

        // Re-read from disk: only that one byte changed; size still the same.
        let new_bytes = read_all(&path);
        assert_eq!(new_bytes.len() as u64, orig_len);

        let mut expected = original.clone();
        expected[10] ^= 0xFF;
        assert_eq!(new_bytes, expected);

        // Ensure exactly one byte differs.
        let diff_count = expected
            .iter()
            .zip(original.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(diff_count, 1, "only one byte should have changed");

        let _ = fs::remove_file(path);
    }
}

/// Removes a file if there's one: its readers (memory maps, open handles) keep it until they're
/// done with it, and a file created in its place is a new one.
pub(crate) fn remove_if_exists<P: AsRef<std::path::Path>>(path: P) -> std::io::Result<()> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Creates a new, empty file at `path`, in place of any there: readers of the one there keep it.
/// Never opens another's: if another file appears at `path` between removing the old one and
/// creating this one (another writer's), that's removed in turn, until this one is created.
pub(crate) fn create_new_file<P: AsRef<std::path::Path>>(
    path: P,
) -> std::io::Result<std::fs::File> {
    loop {
        remove_if_exists(path.as_ref())?;
        match OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(path.as_ref())
        {
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            file => return file,
        }
    }
}
