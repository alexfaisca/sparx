use super::*;

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType, Dir: crate::graph::Directedness>
    GraphMemoryMap<N, E, Ix, Dir>
{
    // /// Export the [`GraphMemoryMap`] instance to petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format keeping all edge and node labelings[^1].
    // ///
    // /// [^1]: if none of the edge or node labeling is wanted consider using [`export_petgraph_stripped`].
    // ///
    // /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    // /// [`export_petgraph_stripped`]: ./struct.GraphMemoryMap.html#method.export_petgraph_stripped
    // pub fn export_petgraph_impl(
    //     &self,
    // ) -> Result<
    //     petgraph::graph::DiGraph<petgraph::graph::NodeIndex<usize>, EdgeType>,
    //     Box<dyn std::error::Error>,
    // > {
    //     let mut graph =
    //         petgraph::graph::DiGraph::<petgraph::graph::NodeIndex<usize>, EdgeType>::new();
    //     let node_count = self.size();
    //
    //     (0..node_count).for_each(|u| {
    //         graph.add_node(petgraph::graph::NodeIndex::new(u));
    //     });
    //     (0..node_count)
    //         .filter_map(|u| match self.neighbours(u) {
    //             Ok(neighbours_of_u) => Some((u, neighbours_of_u)),
    //             Err(e) => {
    //                 eprint!("error getting neihghbours of {u} (proceeding anyways): {e}");
    //                 None
    //             }
    //         })
    //         .for_each(|(u, u_n)| {
    //             u_n.for_each(|v| {
    //                 graph.add_edge(
    //                     petgraph::graph::NodeIndex::new(u),
    //                     petgraph::graph::NodeIndex::new(v.dest()),
    //                     v.e_type(),
    //                 );
    //             });
    //         });
    //
    //     Ok(graph)
    // }

    /// Export the [`GraphMemoryMap`] instance to petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format stripping any edge or node labelings whatsoever.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    pub fn export_petgraph_stripped_impl(
        &self,
    ) -> Result<petgraph::graph::DiGraph<(), ()>, Box<dyn std::error::Error>> {
        let mut graph = petgraph::graph::DiGraph::<(), ()>::new();
        let node_count = self.size();

        (0..node_count).for_each(|_| {
            graph.add_node(());
        });
        (0..node_count)
            .filter_map(|u| match self.neighbours(u) {
                Ok(neighbours_of_u) => Some((u, neighbours_of_u)),
                Err(e) => {
                    eprint!("error getting neihghbours of {u} (proceeding anyways): {e}");
                    None
                }
            })
            .for_each(|(u, u_n)| {
                u_n.for_each(|v| {
                    graph.add_edge(
                        petgraph::graph::NodeIndex::new(u),
                        petgraph::graph::NodeIndex::new(v),
                        (),
                    );
                });
            });

        Ok(graph)
    }
}
