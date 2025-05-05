use std::sync::OnceLock;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::graph;
use rustc_data_structures::graph::dominators::{Dominators, dominators};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_index::bit_set::DenseBitSet;
use rustc_index::{IndexSlice, IndexVec};
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};
use smallvec::SmallVec;

use crate::mir::traversal::Postorder;
use crate::mir::{BasicBlock, BasicBlockData, START_BLOCK, Terminator, TerminatorKind};

#[derive(Clone, TyEncodable, TyDecodable, Debug, HashStable, TypeFoldable, TypeVisitable)]
pub struct BasicBlocks<'tcx> {
    basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>,
    cache: Cache,
}

// Typically 95%+ of basic blocks have 4 or fewer predecessors.
type Predecessors = IndexVec<BasicBlock, DenseBitSet<BasicBlock>>;

/// Each `(target, switch)` entry in the map contains a list of switch values
/// that lead to a `target` block from a `switch` block.
///
/// Note: this type is currently never instantiated, because it's only used for
/// `BasicBlocks::switch_sources`, which is only called by backwards analyses
/// that do `SwitchInt` handling, and we don't have any of those, not even in
/// tests. See #95120 and #94576.
type SwitchSources = FxHashMap<(BasicBlock, BasicBlock), SmallVec<[SwitchTargetValue; 1]>>;

#[derive(Debug, Clone, Copy)]
pub enum SwitchTargetValue {
    // A normal switch value.
    Normal(u128),
    // The final "otherwise" fallback value.
    Otherwise,
}

#[derive(Clone, Default, Debug)]
struct Cache {
    /// For every block we store the immediate predecessors.
    ///
    /// ```text
    ///       a
    ///      / \
    ///     b   c
    ///      \ /
    ///       d
    /// ```
    /// In this case we have:
    /// ```
    /// a: {}
    /// b: {a}
    /// c: {a}
    /// d: {b, c}
    /// ```
    // FIXME: This is equivalent to `BasicBlocks.predecessors` but uses bit sets instead of
    // `SmallVec`. Maybe that should be replaced by this.
    adjacent_predecessors: OnceLock<Predecessors>,

    /// For every block, we store a set of all proceeding blocks.
    ///
    /// ```
    ///       a
    ///      / \
    ///     b   c
    ///      \ /
    ///       d
    /// ```
    /// In this case we have:
    /// ```
    /// a: {}
    /// b: {a}
    /// c: {a}
    /// d: {a, b, c}
    /// ```
    transitive_predecessors: OnceLock<Predecessors>,

    switch_sources: OnceLock<SwitchSources>,
    reverse_postorder: OnceLock<Vec<BasicBlock>>,
    dominators: OnceLock<Dominators<BasicBlock>>,
}

impl<'tcx> BasicBlocks<'tcx> {
    #[inline]
    pub fn new(basic_blocks: IndexVec<BasicBlock, BasicBlockData<'tcx>>) -> Self {
        BasicBlocks { basic_blocks, cache: Cache::default() }
    }

    pub fn dominators(&self) -> &Dominators<BasicBlock> {
        self.cache.dominators.get_or_init(|| dominators(self))
    }

    /// Returns predecessors for each basic block.
    #[inline]
    pub fn predecessors(&self) -> &Predecessors {
        self.cache.adjacent_predecessors.get_or_init(|| {
            let mut preds = IndexVec::from_elem_n(DenseBitSet::new_empty(self.len()), self.len());
            for (bb, data) in self.basic_blocks.iter_enumerated() {
                if let Some(term) = &data.terminator {
                    for succ in term.successors() {
                        preds[succ].insert(bb);
                    }
                }
            }
            preds
        })
    }

    /// For every block, we store a set of all proceeding blocks.
    ///
    /// ```
    ///       a
    ///      / \
    ///     b   c
    ///      \ /
    ///       d
    /// ```
    /// In this case we have:
    /// ```
    /// a: {}
    /// b: {a}
    /// c: {a}
    /// d: {a, b, c}
    /// ```
    pub fn transitive_predecessors(&self) -> &Predecessors {
        self.cache.transitive_predecessors.get_or_init(|| {
            // Compute `transitive_predecessors`
            let mut transitive_predecessors =
                IndexVec::from_elem_n(DenseBitSet::new_empty(self.len()), self.len());
            // The stack is initially a reversed postorder traversal of the CFG. However, we might add
            // add blocks again to the stack if we have loops.
            let mut stack = self.reverse_postorder().iter().rev().copied().collect::<Vec<_>>();
            // We keep track of all blocks that are currently not in the stack.
            let mut not_in_stack = DenseBitSet::new_empty(self.len());
            while let Some(block) = stack.pop() {
                not_in_stack.insert(block);

                // Loop over all successors to the block and add `block` to their predecessors.
                for succ_block in self[block].terminator().successors() {
                    // Keep track of whether the transitive predecessors of `succ_block` has changed.
                    let mut changed = false;

                    changed |= transitive_predecessors[succ_block].insert(block);

                    // Add all transitive predecessors of `block` to the transitive predecessors of
                    // `succ_block`.
                    if block != succ_block {
                        let (blocks_predecessors, succ_blocks_predecessors) =
                            transitive_predecessors.pick2_mut(block, succ_block);
                        changed |= succ_blocks_predecessors.union(blocks_predecessors);

                        // Check if the `succ_block`s transitive predecessors changed. If so, we may
                        // need to add it to the stack again.
                        if changed && not_in_stack.remove(succ_block) {
                            stack.push(succ_block);
                        }
                    }
                }
            }

            transitive_predecessors
        })
    }

    /// Returns basic blocks in a reverse postorder.
    ///
    /// See [`traversal::reverse_postorder`]'s docs to learn what is preorder traversal.
    ///
    /// [`traversal::reverse_postorder`]: crate::mir::traversal::reverse_postorder
    #[inline]
    pub fn reverse_postorder(&self) -> &[BasicBlock] {
        self.cache.reverse_postorder.get_or_init(|| {
            let mut rpo: Vec<_> = Postorder::new(&self.basic_blocks, START_BLOCK, None).collect();
            rpo.reverse();
            rpo
        })
    }

    /// Returns info about switch values that lead from one block to another
    /// block. See `SwitchSources`.
    #[inline]
    pub fn switch_sources(&self) -> &SwitchSources {
        self.cache.switch_sources.get_or_init(|| {
            let mut switch_sources: SwitchSources = FxHashMap::default();
            for (bb, data) in self.basic_blocks.iter_enumerated() {
                if let Some(Terminator {
                    kind: TerminatorKind::SwitchInt { targets, .. }, ..
                }) = &data.terminator
                {
                    for (value, target) in targets.iter() {
                        switch_sources
                            .entry((target, bb))
                            .or_default()
                            .push(SwitchTargetValue::Normal(value));
                    }
                    switch_sources
                        .entry((targets.otherwise(), bb))
                        .or_default()
                        .push(SwitchTargetValue::Otherwise);
                }
            }
            switch_sources
        })
    }

    /// Returns mutable reference to basic blocks. Invalidates CFG cache.
    #[inline]
    pub fn as_mut(&mut self) -> &mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        self.invalidate_cfg_cache();
        &mut self.basic_blocks
    }

    /// Get mutable access to basic blocks without invalidating the CFG cache.
    ///
    /// By calling this method instead of e.g. [`BasicBlocks::as_mut`] you promise not to change
    /// the CFG. This means that
    ///
    ///  1) The number of basic blocks remains unchanged
    ///  2) The set of successors of each terminator remains unchanged.
    ///  3) For each `TerminatorKind::SwitchInt`, the `targets` remains the same and the terminator
    ///     kind is not changed.
    ///
    /// If any of these conditions cannot be upheld, you should call [`BasicBlocks::invalidate_cfg_cache`].
    #[inline]
    pub fn as_mut_preserves_cfg(&mut self) -> &mut IndexVec<BasicBlock, BasicBlockData<'tcx>> {
        &mut self.basic_blocks
    }

    /// Invalidates cached information about the CFG.
    ///
    /// You will only ever need this if you have also called [`BasicBlocks::as_mut_preserves_cfg`].
    /// All other methods that allow you to mutate the basic blocks also call this method
    /// themselves, thereby avoiding any risk of accidentally cache invalidation.
    pub fn invalidate_cfg_cache(&mut self) {
        self.cache = Cache::default();
    }
}

impl<'tcx> std::ops::Deref for BasicBlocks<'tcx> {
    type Target = IndexSlice<BasicBlock, BasicBlockData<'tcx>>;

    #[inline]
    fn deref(&self) -> &IndexSlice<BasicBlock, BasicBlockData<'tcx>> {
        &self.basic_blocks
    }
}

impl<'tcx> graph::DirectedGraph for BasicBlocks<'tcx> {
    type Node = BasicBlock;

    #[inline]
    fn num_nodes(&self) -> usize {
        self.basic_blocks.len()
    }
}

impl<'tcx> graph::StartNode for BasicBlocks<'tcx> {
    #[inline]
    fn start_node(&self) -> Self::Node {
        START_BLOCK
    }
}

impl<'tcx> graph::Successors for BasicBlocks<'tcx> {
    #[inline]
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.basic_blocks[node].terminator().successors()
    }
}

impl<'tcx> graph::Predecessors for BasicBlocks<'tcx> {
    #[inline]
    fn predecessors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.predecessors()[node].iter()
    }
}

// Done here instead of in `structural_impls.rs` because `Cache` is private, as is `basic_blocks`.
TrivialTypeTraversalImpls! { Cache }

impl<S: Encoder> Encodable<S> for Cache {
    #[inline]
    fn encode(&self, _s: &mut S) {}
}

impl<D: Decoder> Decodable<D> for Cache {
    #[inline]
    fn decode(_: &mut D) -> Self {
        Default::default()
    }
}

impl<CTX> HashStable<CTX> for Cache {
    #[inline]
    fn hash_stable(&self, _: &mut CTX, _: &mut StableHasher) {}
}
