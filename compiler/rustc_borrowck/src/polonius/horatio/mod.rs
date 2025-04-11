#![allow(dead_code)]
#![deny(unused_imports)]
mod constraints;
mod live_region_variance;
mod loan_invalidations;
use std::cell::OnceCell;
use std::sync::LazyLock;

use constraints::{Constraints, TimeTravellingRegions};
use itertools::Either;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::bit_set::thin_bit_set::{SparseBitMatrix, ThinBitSet};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::{
    self, BasicBlock, BasicBlockData, Body, Local, Location, Place, Statement, Terminator,
};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::points::DenseLocationMap;
use smallvec::{SmallVec, smallvec};

use super::ConstraintDirection;
use crate::{
    BorrowData, BorrowIndex, BorrowSet, PlaceConflictBias, PlaceExt, RegionInferenceContext,
    RegionVid, places_conflict,
};

/// This toggles the `my_println!` and `my_print!` macros. Those macros are used here and there to
/// print tracing information about Polonius.
pub(crate) const MY_DEBUG_PRINTS: LazyLock<bool> = LazyLock::new(|| {
    matches!(std::env::var("POLONIUS_TRACING").as_ref().map(String::as_str), Ok("1"))
});

macro_rules! my_println {
    ($($x:expr),*) => {
        if *crate::polonius::horatio::MY_DEBUG_PRINTS {
            println!($($x,)*);
        }
    };
}
pub(crate) use my_println;

macro_rules! my_print {
    ($($x:expr),*) => {
        if *crate::polonius::horatio::MY_DEBUG_PRINTS {
            print!($($x,)*);
        }
    };
}
pub(crate) use my_print;

/// A cache remembering whether a loan is killed at a block.
type KillsCache = IndexVec<PoloniusBlock, Option<KillAtBlock>>;

pub(crate) struct PoloniusOutOfScopePrecomputer<'a, 'tcx> {
    pub pcx: PoloniusContext<'a, 'tcx>,
    borrows: IndexVec<BorrowIndex, Option<PoloniusBorrowData>>,
}

pub(crate) struct PoloniusContext<'a, 'tcx> {
    cache: OnceCell<Cache<'a, 'tcx>>,

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
    transitive_predecessors: IndexVec<BasicBlock, ThinBitSet<BasicBlock>>,

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
    adjacent_predecessors: IndexVec<BasicBlock, ThinBitSet<BasicBlock>>,

    /// Only computed for diagnostics: The regions that outlive free regions are used to distinguish
    /// relevant live locals from boring locals. A boring local is one whose type contains only such
    /// regions. Polonius currently has more boring locals than NLLs so we record the latter to use
    /// in errors and diagnostics, to focus on the locals we consider relevant and match NLL
    /// diagnostics.
    boring_nll_locals: OnceCell<ThinBitSet<Local>>,

    tcx: TyCtxt<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    body: &'a Body<'tcx>,
    location_map: &'a DenseLocationMap,
    borrow_set: &'a BorrowSet<'tcx>,
}

struct Cache<'a, 'tcx> {
    /// All universal regions.
    universal_regions: ThinBitSet<RegionVid>,

    /// All outlives constraints.
    constraints: Constraints<'a, 'tcx>,
}

#[derive(Copy, Clone)]
struct BorrowContext<'a, 'b, 'tcx> {
    pcx: &'a PoloniusContext<'b, 'tcx>,
    borrow_idx: BorrowIndex,
    borrow: &'a BorrowData<'tcx>,
}

enum PoloniusBorrowData {
    /// This borrow should be ignored.
    Ignored,

    Data {
        kills_cache: KillsCache,
        possibly_dependent_regions: Option<ThinBitSet<RegionVid>>,
        scope_computation: Option<ScopeComputation>,
    },
}

rustc_index::newtype_index! {
    /// A `PoloniusBlock` is a `BasicBlock` which splits the block where a loan is introduced into
    /// two blocks.
    ///
    /// The problem is that we want to record at most one location per block where a loan is killed.
    /// But a loan might be killed twice in the block where it is introduced, both before and after
    /// the reserve location. So we use an additional index to denote the introduction block up to
    /// and including the statement where the loan is introduced. This has the consequence that a
    /// `PoloniusBlock` is specific for a given loan.
    ///
    /// We call the block containing all statements after the reserve location for the
    /// "introduction block", and the block containing statements up to and including the reserve
    /// location "before introduction block". These names might be bad, but my (Tage's) fantacy
    /// struggles to come up with anything better.
    ///
    /// So if the loan is introduced at `bb2[2]`, `bb2[0..=2]` is the "before introduction block"
    /// and `bb2[3..]` is the "introduction block".
    ///
    /// For a given loan `l` introduced at a basic block `b`, a `PoloniusBlock` is equivalent to a
    /// `BasicBlocka with the following exceptions:
    /// - `PoloniusBlock::from_u32(b.as_u32())` is `l`'s introduction block.
    /// - `PoloniusBlock::from_usize(basic_blocks.len())` is `l`'s "before introduction block".
    #[debug_format = "pbb{}"]
    pub struct PoloniusBlock {}
}

impl PoloniusBlock {
    /// Converts a [`BasicBlock`] to a [`PoloniusBlock`] assuming this is not the "before
    /// introduction block".
    #[inline]
    fn from_basic_block(basic_block: BasicBlock) -> Self {
        Self::from_u32(basic_block.as_u32())
    }

    /// Get the "introduction block". I.E the first block where the loan is introduced.
    #[inline]
    fn introduction_block(bcx: BorrowContext<'_, '_, '_>) -> Self {
        Self::from_basic_block(bcx.borrow.reserve_location.block)
    }

    /// Get the "before introduction block". I.E the block consisting of statements up to and
    /// including the loan's reserve location.
    #[inline]
    fn before_introduction_block(bcx: BorrowContext<'_, '_, '_>) -> Self {
        Self::from_usize(bcx.pcx.body.basic_blocks.len())
    }

    /// Get the correct block from a loan and a location.
    #[inline]
    fn from_location(bcx: BorrowContext<'_, '_, '_>, location: Location) -> Self {
        if location.block == bcx.borrow.reserve_location.block
            && location.statement_index <= bcx.borrow.reserve_location.statement_index
        {
            Self::before_introduction_block(bcx)
        } else {
            Self::from_basic_block(location.block)
        }
    }

    /// Returns the number of polonius blocks. THat is, the number of blocks + 1.
    #[inline]
    fn num_blocks(bcx: BorrowContext<'_, '_, '_>) -> usize {
        bcx.pcx.body.basic_blocks.len() + 1
    }

    /// Get the [`BasicBlock`] containing this [`PoloniusBlock``].
    #[inline]
    fn basic_block(self, bcx: BorrowContext<'_, '_, '_>) -> BasicBlock {
        if self.as_usize() == bcx.pcx.body.basic_blocks.len() {
            bcx.borrow.reserve_location.block
        } else {
            BasicBlock::from_u32(self.as_u32())
        }
    }

    /// Check if this is the "introduction block". I.E the block immediately after the loan has been
    /// introduced.
    #[inline]
    fn is_introduction_block(self, bcx: BorrowContext<'_, '_, '_>) -> bool {
        self.as_u32() == bcx.borrow.reserve_location.block.as_u32()
    }

    /// Check if this is the "before introduction block". I.E the block containing statements up to
    /// and including the loan's reserve location.
    #[inline]
    fn is_before_introduction_block(self, bcx: BorrowContext<'_, '_, '_>) -> bool {
        self.as_usize() == bcx.pcx.body.basic_blocks.len()
    }

    /// Get the index of the first statement in this block. This will be 0 except for the
    /// introduction block.
    #[inline]
    fn first_index(self, bcx: BorrowContext<'_, '_, '_>) -> usize {
        if self.is_introduction_block(bcx) {
            bcx.borrow.reserve_location.statement_index + 1
        } else {
            0
        }
    }

    /// Get the last statement index for this block. For all blocks except the "before introduction
    /// block", this will point to a terminator, not a statement.
    #[inline]
    fn last_index(self, bcx: BorrowContext<'_, '_, '_>) -> usize {
        if !self.is_before_introduction_block(bcx) {
            bcx.pcx.body.basic_blocks[self.basic_block(bcx)].statements.len()
        } else {
            bcx.borrow.reserve_location.statement_index
        }
    }

    /// Iterate over the successor blocks to this block.
    ///
    /// Note that this is same as [`Terminator::successors`] except for the "before introduction
    /// block" where it is the "introduction block".
    #[inline]
    fn successors(
        self,
        bcx: BorrowContext<'_, '_, '_>,
    ) -> impl DoubleEndedIterator<Item = PoloniusBlock> {
        if !self.is_before_introduction_block(bcx) {
            Either::Left(bcx.pcx.body[self.basic_block(bcx)].terminator().successors().map(
                move |bb| {
                    if bb == bcx.borrow.reserve_location.block {
                        Self::before_introduction_block(bcx)
                    } else {
                        Self::from_basic_block(bb)
                    }
                },
            ))
        } else {
            Either::Right([Self::introduction_block(bcx)].into_iter())
        }
    }
}

struct LoanRegionNode {
    associated_regions: ThinBitSet<RegionVid>,
    added_regions: Option<ThinBitSet<RegionVid>>,
    /// Whether this location is reachable by forward edges from the loan's introduction point in
    /// the localized constraint graph.
    reachable_by_loan: bool,
    /// Whether the loan is in scope at this point.
    in_scope: bool,
    /// Whether this node has been added to the stack for processing.
    added_to_stack: bool,
}

struct ScopeComputation {
    nodes: FxHashMap<Location, LoanRegionNode>,
    primary_stack: Vec<Location>,
    secondary_stack: Vec<Location>,
    is_finished: bool,
}

/// Information of when/if a loan is killed at a block.
#[derive(Debug, Copy, Clone)]
enum KillAtBlock {
    /// The loan is not killed at this block.
    NotKilled,

    /// The loan is killed.
    Killed { statement_index: usize },
}
use KillAtBlock::*;

impl<'a, 'tcx> PoloniusContext<'a, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        regioncx: &'a RegionInferenceContext<'tcx>,
        body: &'a Body<'tcx>,
        location_map: &'a DenseLocationMap,
        borrow_set: &'a BorrowSet<'tcx>,
    ) -> Self {
        // Compute `transitive_predecessors` and `adjacent_predecessors`.
        let mut transitive_predecessors = IndexVec::from_elem_n(
            ThinBitSet::new_empty(body.basic_blocks.len()),
            body.basic_blocks.len(),
        );
        let mut adjacent_predecessors = transitive_predecessors.clone();
        // The stack is initially a reversed postorder traversal of the CFG. However, we might add
        // add blocks again to the stack if we have loops.
        let mut stack =
            body.basic_blocks.reverse_postorder().iter().rev().copied().collect::<Vec<_>>();
        // We keep track of all blocks that are currently not in the stack.
        let mut not_in_stack = ThinBitSet::new_empty(body.basic_blocks.len());
        while let Some(block) = stack.pop() {
            not_in_stack.insert(block);

            // Loop over all successors to the block and add `block` to their predecessors.
            for succ_block in body.basic_blocks[block].terminator().successors() {
                // Keep track of whether the transitive predecessors of `succ_block` has changed.
                let mut changed = false;

                // Insert `block` in `succ_block`s predecessors.
                if adjacent_predecessors[succ_block].insert(block) {
                    // Remember that `adjacent_predecessors` is a subset of
                    // `transitive_predecessors`.
                    changed |= transitive_predecessors[succ_block].insert(block);
                }

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

            debug_assert!(transitive_predecessors[block].superset(&adjacent_predecessors[block]));
        }

        Self {
            cache: OnceCell::new(),
            transitive_predecessors,
            adjacent_predecessors,
            boring_nll_locals: OnceCell::new(),
            tcx,
            regioncx,
            body,
            location_map,
            borrow_set,
        }
    }

    fn cache(&self) -> &Cache<'a, 'tcx> {
        self.cache.get_or_init(|| {
            let mut universal_regions = new_empty_region_set(self.regioncx);
            universal_regions
                .insert_range(self.regioncx.universal_regions().universal_regions_range());

            let mut constraints =
                Constraints::new(self.tcx, self.regioncx, self.body, self.location_map);
            for constraint in self.regioncx.outlives_constraints() {
                constraints.add_constraint(&constraint);
            }

            Cache { universal_regions, constraints }
        })
    }

    fn boring_nll_locals(&self) -> &ThinBitSet<Local> {
        self.boring_nll_locals.get_or_init(|| {
            let mut free_regions = new_empty_region_set(self.regioncx);
            for region in self.regioncx.universal_regions().universal_regions_iter() {
                free_regions.insert(region);
            }
            self.cache().constraints.add_dependent_regions_reversed(&mut free_regions);

            let mut boring_locals = ThinBitSet::new_empty(self.body.local_decls.len());
            for (local, local_decl) in self.body.local_decls.iter_enumerated() {
                if self
                    .tcx
                    .all_free_regions_meet(&local_decl.ty, |r| free_regions.contains(r.as_var()))
                {
                    boring_locals.insert(local);
                }
            }

            boring_locals
        })
    }

    pub(crate) fn is_boring_local(&self, local: Local) -> bool {
        self.boring_nll_locals().contains(local)
    }

    /// Returns `true` iff `a` is earlier in the control flow graph than `b`.
    #[inline]
    fn is_predecessor(&self, a: Location, b: Location) -> bool {
        a.block == b.block && a.statement_index < b.statement_index
            || self.transitive_predecessors[b.block].contains(a.block)
    }
}

impl<'a, 'b, 'tcx> BorrowContext<'a, 'b, 'tcx> {
    /// Construct a new empty set with capacity for [`PoloniusBlock`]s.
    fn new_polonius_block_set(self) -> ThinBitSet<PoloniusBlock> {
        ThinBitSet::new_empty(PoloniusBlock::num_blocks(self))
    }
}

impl<'a, 'tcx> PoloniusOutOfScopePrecomputer<'a, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        regioncx: &'a RegionInferenceContext<'tcx>,
        body: &'a Body<'tcx>,
        location_map: &'a DenseLocationMap,
        borrow_set: &'a BorrowSet<'tcx>,
    ) -> Self {
        Self {
            pcx: PoloniusContext::new(tcx, regioncx, body, location_map, borrow_set),
            borrows: IndexVec::new(),
        }
    }

    /// Quick check to check if the loan is in scope.
    pub(crate) fn loan_maybe_in_scope_at(
        &mut self,
        borrow_idx: BorrowIndex,
        borrow: &BorrowData<'tcx>,
        location: Location,
    ) -> bool {
        // Check if this location can never be reached by the borrow.
        if !self.pcx.is_predecessor(borrow.reserve_location(), location) {
            return false;
        }

        let maybe_borrow_data = self.borrows.ensure_contains_elem(borrow_idx, || None);
        match maybe_borrow_data {
            Some(PoloniusBorrowData::Ignored) => return false,
            Some(PoloniusBorrowData::Data {
                scope_computation: Some(scope_computation), ..
            }) => {
                // Check if we have already computed an "in scope-value" for location.
                if scope_computation.is_finished {
                    // If the scope computation is finished, it's appropriate to return `false` if no
                    // node for the location exists.
                    return scope_computation.nodes.get(&location).is_some_and(|x| x.in_scope);

                    // If the computation is not finished, we can only be sure if the `in_scope`-field
                    // has been set to `true` for the relevant node.
                } else if scope_computation.nodes.get(&location).is_some_and(|x| x.in_scope) {
                    return true;
                }
            }
            None => {
                // Check if this borrow is ignored.
                if borrow.borrowed_place().ignore_borrow(
                    self.pcx.tcx,
                    self.pcx.body,
                    &self.pcx.borrow_set.locals_state_at_exit,
                ) {
                    *maybe_borrow_data = Some(PoloniusBorrowData::Ignored);
                    return false;
                }
            }
            Some(PoloniusBorrowData::Data { scope_computation: None, .. }) => (),
        };

        /*
        if !self.pcx.regioncx.region_contains(borrow.region, location) {
            return false;
        }
        */

        return true;
    }

    /// Check if a loan is in scope at a location.
    pub(crate) fn loan_in_scope_at(
        &mut self,
        borrow_idx: BorrowIndex,
        borrow: &BorrowData<'tcx>,
        location: Location,
    ) -> bool {
        let maybe_borrow_data = &mut self.borrows[borrow_idx];
        match maybe_borrow_data {
            Some(PoloniusBorrowData::Ignored) => unreachable!(),
            Some(PoloniusBorrowData::Data {
                scope_computation: Some(scope_computation), ..
            }) => {
                // Check if we have already computed an "in scope-value" for location.
                if scope_computation.is_finished {
                    // If the scope computation is finished, it's appropriate to return `false` if no
                    // node for the location exists.
                    return scope_computation.nodes.get(&location).is_some_and(|x| x.in_scope);

                    // If the computation is not finished, we can only be sure if the `in_scope`-field
                    // has been set to `true` for the relevant node.
                } else if scope_computation.nodes.get(&location).is_some_and(|x| x.in_scope) {
                    return true;
                }
            }
            Some(PoloniusBorrowData::Data { .. }) => (),
            None => {
                *maybe_borrow_data = Some(PoloniusBorrowData::Data {
                    kills_cache: IndexVec::new(),
                    scope_computation: None,
                    possibly_dependent_regions: None,
                });
            }
        };

        let Some(PoloniusBorrowData::Data { kills_cache, scope_computation, .. }) =
            maybe_borrow_data
        else {
            unreachable!()
        };

        let bcx = BorrowContext { pcx: &self.pcx, borrow_idx, borrow };

        // Check if the loan is killed anywhere between its reserve location and `location`.
        let Some(live_paths) = live_paths(bcx, kills_cache, location) else {
            return false;
        };

        scope_computation.get_or_insert_with(|| ScopeComputation::new(bcx)).compute(
            bcx,
            kills_cache,
            location,
            live_paths,
        )
    }
}

/// Returns `true` if the loan is killed at `location`. Note that the kill takes effect at the next
/// statement.
fn is_killed(
    bcx: BorrowContext<'_, '_, '_>,
    kills_cache: &mut KillsCache,
    location: Location,
) -> bool {
    let polonius_block = PoloniusBlock::from_location(bcx, location);

    // Check if we already know the answer.
    match kills_cache.get(polonius_block) {
        Some(Some(Killed { statement_index })) => {
            return *statement_index == location.statement_index;
        }
        Some(Some(NotKilled)) => return false,
        Some(None) | None => (),
    }
    // The answer was not known so we have to compute it ourselfs.

    let is_kill =
        if let Some(stmt) = bcx.pcx.body[location.block].statements.get(location.statement_index) {
            is_killed_at_stmt(bcx, stmt)
        } else {
            is_killed_at_terminator(bcx, &bcx.pcx.body[location.block].terminator())
        };

    // If we had a kill at this location, we should add it to the cache.
    if is_kill {
        *kills_cache.ensure_contains_elem(polonius_block, || None) =
            Some(Killed { statement_index: location.statement_index });
    }

    is_kill
}

/// Calculate when/if a loan goes out of scope for a set of statements in a block.
fn is_killed_at_block(
    bcx: BorrowContext<'_, '_, '_>,
    kills_cache: &mut KillsCache,
    block: PoloniusBlock,
) -> bool {
    let res = kills_cache.get_or_insert_with(block, || {
        let block_data = &bcx.pcx.body[block.basic_block(bcx)];
        for statement_index in block.first_index(bcx)..=block.last_index(bcx) {
            let is_killed = if let Some(stmt) = block_data.statements.get(statement_index) {
                is_killed_at_stmt(bcx, stmt)
            } else {
                is_killed_at_terminator(bcx, &block_data.terminator())
            };

            if is_killed {
                return Killed { statement_index };
            }
        }

        NotKilled
    });

    matches!(res, Killed { .. })
}

/// Given that the borrow was in scope on entry to this statement, check if it goes out of scope
/// till the next location.
#[inline]
fn is_killed_at_stmt<'tcx>(bcx: BorrowContext<'_, '_, 'tcx>, stmt: &Statement<'tcx>) -> bool {
    match &stmt.kind {
        mir::StatementKind::Assign(box (lhs, _rhs)) => kill_on_place(bcx, *lhs),
        mir::StatementKind::StorageDead(local) => {
            bcx.pcx.borrow_set.local_map.get(local).is_some_and(|bs| bs.contains(&bcx.borrow_idx))
        }
        _ => false,
    }
}

/// Given that the borrow was in scope on entry to this terminator, check if it goes out of scope
/// till the succeeding blocks.
#[inline]
fn is_killed_at_terminator<'tcx>(
    bcx: BorrowContext<'_, '_, 'tcx>,
    terminator: &Terminator<'tcx>,
) -> bool {
    match &terminator.kind {
        // A `Call` terminator's return value can be a local which has borrows, so we need to record
        // those as killed as well.
        mir::TerminatorKind::Call { destination, .. } => kill_on_place(bcx, *destination),
        mir::TerminatorKind::InlineAsm { operands, .. } => operands.iter().any(|op| {
            if let mir::InlineAsmOperand::Out { place: Some(place), .. }
            | mir::InlineAsmOperand::InOut { out_place: Some(place), .. } = op
            {
                kill_on_place(bcx, *place)
            } else {
                false
            }
        }),
        _ => false,
    }
}

#[inline]
fn kill_on_place<'tcx>(bcx: BorrowContext<'_, '_, 'tcx>, place: Place<'tcx>) -> bool {
    bcx.pcx.borrow_set.local_map.get(&place.local).is_some_and(|bs| bs.contains(&bcx.borrow_idx))
        && if place.projection.is_empty() {
            !bcx.pcx.body.local_decls[place.local].is_ref_to_static()
        } else {
            places_conflict(
                bcx.pcx.tcx,
                bcx.pcx.body,
                bcx.borrow.borrowed_place,
                place,
                PlaceConflictBias::NoOverlap,
            )
        }
}

/// Remove dead regions from the set of associated regions.
fn remove_dead_regions(
    pcx: &PoloniusContext<'_, '_>,
    location: Location,
    region_set: &mut ThinBitSet<RegionVid>,
) {
    for region in region_set.clone().iter() {
        if !pcx.regioncx.liveness_constraints().is_live_at(region, location) {
            region_set.remove(region);
        }
    }
}

impl ScopeComputation {
    fn new(bcx: BorrowContext<'_, '_, '_>) -> Self {
        // Put the loan's initial region in a set.
        let mut initial_region_set = new_empty_region_set(bcx.pcx.regioncx);
        initial_region_set.insert(bcx.borrow.region);

        let mut nodes = FxHashMap::default();
        nodes.insert(
            bcx.borrow.reserve_location,
            LoanRegionNode {
                associated_regions: new_empty_region_set(bcx.pcx.regioncx),
                added_regions: Some(initial_region_set),
                reachable_by_loan: false,
                in_scope: false,
                added_to_stack: true,
            },
        );
        Self {
            primary_stack: vec![bcx.borrow.reserve_location],
            secondary_stack: vec![],
            nodes,
            is_finished: false,
        }
    }

    #[inline(never)] // FIXME: Remove this.
    fn compute(
        &mut self,
        bcx: BorrowContext<'_, '_, '_>,
        kills_cache: &mut KillsCache,
        target_location: Location,
        live_paths: ThinBitSet<PoloniusBlock>,
    ) -> bool {
        my_println!("Checking {:?} at {:?}", bcx.borrow_idx, target_location);
        debug_assert!(!self.is_finished);

        while let Some(location) = self.primary_stack.pop().or_else(|| self.secondary_stack.pop()) {
            let point = bcx.pcx.location_map.point_from_location(location);
            let block_data = &bcx.pcx.body[location.block];

            // Debugging: Print the current location and statement/expression.
            if let Some(stmt) = block_data.statements.get(location.statement_index) {
                my_println!("  {:?}: {:?}", location, stmt);
            } else {
                my_println!("  {:?}: {:?}", location, block_data.terminator().kind);
            }

            // Fetch the current node.
            let LoanRegionNode {
                associated_regions,
                added_regions,
                reachable_by_loan,
                in_scope,
                added_to_stack,
            } = self.nodes.get_mut(&location).unwrap();
            let reachable_by_loan = *reachable_by_loan; // Make copy.

            debug_assert!(*added_to_stack);
            *added_to_stack = false;

            let time_travelling_regions = if let Some(mut added_regions) = added_regions.take() {
                debug_assert!(!added_regions.is_empty(), "added_regions should never be empty.");
                debug_assert!(
                    added_regions.iter().all(|r| !associated_regions.contains(r)),
                    "added_regions and associated_regions should be disjunct."
                );

                // Add constraints.
                let time_travelling_regions = bcx
                    .pcx
                    .cache()
                    .constraints
                    .add_dependent_regions_at_point(point, &mut added_regions);
                if let Some(tf) = &time_travelling_regions.to_next_loc {
                    my_println!("    Forward time travellers: {:?}", tf);
                }
                if let Some(tf) = &time_travelling_regions.to_prev_stmt {
                    my_println!("    Backward time travellers: {:?}", tf);
                }
                if let Some(x) = &time_travelling_regions.to_predecessor_blocks {
                    my_println!("    To preceeding blocks: {:?}", x);
                }
                if let Some(x) = &time_travelling_regions.to_successor_blocks {
                    my_println!("    To succeeding blocks: {:?}", x);
                }

                // Incorporate the added regions into `associated_regions`.
                associated_regions.union(&added_regions);
                my_println!("    Regions: {:?}", associated_regions);

                Some(time_travelling_regions)
            } else {
                my_println!("Nothing new here.");
                // FIXME: This should be unnecessary if we don't track kills.
                if reachable_by_loan {
                    // FIXME: This is just a hack.
                    let mut associated_regions = associated_regions.clone();
                    remove_dead_regions(bcx.pcx, location, &mut associated_regions);
                    if associated_regions.is_empty() {
                        my_println!("  Loan killed.");
                        continue;
                    }
                } else {
                    continue;
                }

                None
            };

            let mut associated_regions = associated_regions.clone();
            remove_dead_regions(bcx.pcx, location, &mut associated_regions);
            if reachable_by_loan && !associated_regions.is_empty() {
                *in_scope = true;
                my_println!("    In scope at {location:?}");
            }

            // Check if the loan is killed.
            let is_killed = is_killed(bcx, kills_cache, location);

            if is_killed && bcx.pcx.is_predecessor(bcx.borrow.reserve_location, location) {
                continue;
            }

            let successor_reachable_by_loan =
                !is_killed && reachable_by_loan || location == bcx.borrow.reserve_location;

            // Necessary to make the borrow checker happy.
            let in_scope = *in_scope;

            visit_adjacent_locations(
                bcx.pcx,
                block_data,
                location,
                time_travelling_regions,
                |new_location, time_travellers, is_forward| {
                    let new_node =
                        self.nodes.entry(new_location).or_insert_with(|| LoanRegionNode {
                            associated_regions: new_empty_region_set(bcx.pcx.regioncx),
                            added_regions: None,
                            reachable_by_loan: false,
                            in_scope: false,
                            added_to_stack: false,
                        });

                    // Keep track of whether `new_node` has changed.
                    let mut new_node_changed = false;

                    // If we are going forwards, we need to propagate reachability for the loan.
                    if is_forward && successor_reachable_by_loan && !new_node.reachable_by_loan {
                        new_node.reachable_by_loan = true;
                        // `reachable_by_loan` was `false` before on `new_node` but has now been
                        // changed to `true`.
                        new_node_changed = true;
                    }

                    // Check if any regions should be added to `new_node`.
                    let mut added_regions = associated_regions.clone();
                    if !is_forward {
                        added_regions.subtract(&bcx.pcx.cache().universal_regions);
                    }

                    remove_dead_regions(bcx.pcx, new_location, &mut added_regions);

                    if let Some(time_travellers) = time_travellers {
                        added_regions.union(time_travellers);
                    }

                    added_regions.subtract(&new_node.associated_regions);

                    if !added_regions.is_empty() {
                        if let Some(already_added_regions) = new_node.added_regions.as_mut() {
                            already_added_regions.union(&added_regions);
                        } else {
                            new_node.added_regions = Some(added_regions);
                        }
                        new_node_changed = true;
                    }

                    if new_node_changed && !new_node.added_to_stack {
                        if !is_forward
                            || live_paths.contains(PoloniusBlock::from_location(bcx, new_location))
                        {
                            self.primary_stack.push(new_location);
                        } else {
                            self.secondary_stack.push(new_location);
                        }
                        new_node.added_to_stack = true;
                    }
                },
            );

            if in_scope && location == target_location {
                return true;
            }
        }

        self.is_finished = true;
        self.nodes.get(&target_location).is_some_and(|x| x.in_scope)
    }
}

#[inline]
fn visit_adjacent_locations(
    pcx: &PoloniusContext<'_, '_>,
    block_data: &BasicBlockData<'_>,
    location: Location,
    maybe_time_travellers: Option<TimeTravellingRegions>,
    mut op: impl FnMut(Location, Option<&ThinBitSet<RegionVid>>, bool),
) {
    // Forwards:
    if location.statement_index < block_data.statements.len() {
        let successor_location = location.successor_within_block();
        let time_travellers = maybe_time_travellers.as_ref().and_then(|t| t.to_next_loc.as_ref());
        op(successor_location, time_travellers, true);
    } else {
        for successor_block in block_data.terminator().successors() {
            let successor_location = Location { block: successor_block, statement_index: 0 };
            let time_travellers = maybe_time_travellers
                .as_ref()
                .and_then(|t| t.to_successor_blocks.as_ref().and_then(|x| x.row(successor_block)));
            op(successor_location, time_travellers, true);
        }
    }

    // Backwards:
    if location.statement_index > 0 {
        let predecessor_location = location.predecessor_within_block();
        let time_travellers = maybe_time_travellers.as_ref().and_then(|t| t.to_prev_stmt.as_ref());
        op(predecessor_location, time_travellers, false);
    } else {
        for &predecessor_block in &pcx.body.basic_blocks.predecessors()[location.block] {
            let predecessor_location = Location {
                block: predecessor_block,
                statement_index: pcx.body[predecessor_block].statements.len(),
            };
            let time_travellers = maybe_time_travellers.as_ref().and_then(|t| {
                t.to_predecessor_blocks.as_ref().and_then(|x| x.row(predecessor_block))
            });
            op(predecessor_location, time_travellers, false);
        }
    }
}

#[inline(never)] // FIXME: Remove this.
fn live_paths(
    bcx: BorrowContext<'_, '_, '_>,
    kills_cache: &mut KillsCache,
    destination: Location,
) -> Option<ThinBitSet<PoloniusBlock>> {
    // `destination_block` is the `PoloniusBlock` for `destination`.
    let destination_block = PoloniusBlock::from_location(bcx, destination);

    // We begin by checking the relevant statements in `destination_block`.
    // FIXME: Is this the most efficient solution?
    for statement_index in destination_block.first_index(bcx)..destination.statement_index {
        let location = Location { block: destination.block, statement_index };
        if is_killed(bcx, kills_cache, location) {
            return None;
        }
    }

    if destination_block.is_introduction_block(bcx) {
        // We are finished.
        return Some(bcx.new_polonius_block_set());
    }

    // Traverse all blocks between `reserve_location` and `destination` in the CFG and check for
    // kills. If there is no live path from `reserve_location` to `destination`, we no for sure
    // that the loan is dead at `destination`.

    // Keep track of all visited `PoloniusBlock`s.
    let mut visited = bcx.new_polonius_block_set();

    // The stack contains `(block, path)` pairs, where `block` is a `PoloniusBlock` and `path is
    // a set of `PoloniusBlock`s making a path from `reserve_location` to `destination_block`.
    // In this way we can record the live paths.
    let introduction_block = PoloniusBlock::introduction_block(bcx);
    let mut stack: SmallVec<[(PoloniusBlock, ThinBitSet<PoloniusBlock>); 4]> =
        smallvec![(introduction_block, bcx.new_polonius_block_set())];
    visited.insert(introduction_block);

    let mut valid_paths = None;

    while let Some((block, path)) = stack.pop() {
        // Check if the loan is killed in this block.
        if is_killed_at_block(bcx, kills_cache, block) {
            continue;
        }

        // Loop through all successors to `block` and follow those that are predecessors to
        // `destination.block`.
        for successor in block.successors(bcx) {
            let successor_bb = successor.basic_block(bcx);

            if successor == destination_block {
                // We have reached the destination so let's save this path.
                valid_paths.get_or_insert_with(|| bcx.new_polonius_block_set()).union(&path);

                // We continue traversal to record all live paths.
                continue;
            }

            if !visited.insert(successor) {
                continue;
            }

            // Check that `successor` is a predecessor of `destination_block`.
            //
            // Given two `PoloniusBlock`s a and b, then a is a predecessor of b iff
            // `a.basic_block()` is a predecessor of `b.basic_block()`, or a is the "before
            // introduction block" and b is the "introduction block".
            if !bcx.pcx.transitive_predecessors[destination.block].contains(successor_bb)
                || destination_block.is_introduction_block(bcx)
                    && successor.is_before_introduction_block(bcx)
            {
                // `successor` is not a predecessor of `destination_block`.
                continue;
            }

            // Push `successor` to `path`.
            let mut path = path.clone();
            path.insert(successor);
            stack.push((successor, path));
        }
    }

    valid_paths
}

/// Create an empty bit set with capacity for all regions.
fn new_empty_region_set(regioncx: &RegionInferenceContext<'_>) -> ThinBitSet<RegionVid> {
    ThinBitSet::new_empty(regioncx.last_region_vid().map_or(0, |x| x.index() + 1))
}

fn new_region_matrix<R: Idx>(
    regioncx: &RegionInferenceContext<'_>,
) -> SparseBitMatrix<R, RegionVid> {
    SparseBitMatrix::new(num_regions(regioncx))
}

fn num_regions(regioncx: &RegionInferenceContext<'_>) -> usize {
    regioncx.last_region_vid().map_or(0, |x| x.index() + 1)
}

/// FIXME: Just for debugging.
pub(crate) fn format_body_with_borrows<'tcx>(
    body: &Body<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
) -> String {
    let mut res = String::default();
    for (block, block_data) in body.basic_blocks.iter_enumerated() {
        res += &format!("{:?}:\n", block);
        for statement_index in 0..=block_data.statements.len() {
            let location = Location { block, statement_index };
            res += &format!("  {}: ", statement_index);
            if let Some(stmt) = body[location.block].statements.get(location.statement_index) {
                res += &format!("{:?}\n", stmt);
            } else {
                debug_assert_eq!(location.statement_index, body[location.block].statements.len());
                let terminator = body[location.block].terminator();
                res += &format!("{:?}\n", terminator.kind);
            }

            let introduced_borrows = borrow_set
                .iter_enumerated()
                .filter(|(_, b)| b.reserve_location == location)
                .collect::<Vec<_>>();
            if !introduced_borrows.is_empty() {
                res += "    reserved borrows: ";
                for (borrow_idx, _) in introduced_borrows {
                    res += &format!("{:?}, ", borrow_idx);
                }
                res += "\n"
            }
        }
    }
    res
}
