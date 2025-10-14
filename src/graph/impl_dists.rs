use super::GraphMemoryMap;

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> GraphMemoryMap<N, E, Ix> {
    pub(crate) fn degree_dist_impl(&self) -> Result<Box<[usize]>, Box<dyn std::error::Error>> {
        unimplemented!();
        // let node_count = self.size();
        // let threads = self.thread_num().max(get_physical() * 2);
        // let node_load = node_count.div_ceil(threads);
        //
        // let offsets_ptr = SharedSlice::<usize>::new(self.offsets_ptr(), self.offsets_size());
        // let mut degs: Vec<usize> = vec![];
        //
        // thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
        //     // Thread syncronization
        //     let synchronize = Arc::new(Barrier::new(threads));
        //     let mut handles = vec![];
        //
        //     for tid in 0..threads {
        //         let begin = std::cmp::min(tid * node_load, node_count);
        //         let end = std::cmp::min(begin + node_load, node_count);
        //         handles.push(scope.spawn(move |_| {}));
        //     }
        //     // check for errors
        //     for (tid, r) in handles.into_iter().enumerate() {
        //         r.join().map_err(|e| -> Box<dyn std::error::Error> {
        //             format!("error in thread {tid}: {:?}", e).into()
        //         })?;
        //     }
        //     Ok(())
        // })
        // .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        // Ok([].as_slice().into())
    }
}
