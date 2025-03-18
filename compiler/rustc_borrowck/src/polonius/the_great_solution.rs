#![allow(dead_code)]
#![deny(unused_imports)]
mod constraints;
mod loan_invalidations;
use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::sync::LazyLock;

use constraints::Constraints;
use itertools::Either;
use rustc_data_structures::fx::FxHashMap;
use rustc_index::bit_set::thin_bit_set::{SparseBitMatrix, ThinBitSet};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::{self, BasicBlock, Body, Location, Place, Statement, Terminator};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::points::{DenseLocationMap, PointIndex};
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
        if *crate::polonius::the_great_solution::MY_DEBUG_PRINTS {
            println!($($x,)*);
        }
    };
}
pub(crate) use my_println;

macro_rules! my_print {
    ($($x:expr),*) => {
        if *crate::polonius::the_great_solution::MY_DEBUG_PRINTS {
            print!($($x,)*);
        }
    };
}
pub(crate) use my_print;

pub(crate) struct PoloniusOutOfScopePrecomputer<'a, 'tcx> {
    /// A set of the loans that has been checked.
    checked_loans: ThinBitSet<BorrowIndex>,

    /// A cache for remembering which loans should be ignored.
    ///
    /// We have three scenarios:
    /// - A loan is not in `self.checked_loans`: Then we don't know if it should be ignored and we
    ///   need to compute it.
    /// - The loan is in `self.checked_loans` but not in this set: Then the loan should not be
    ///   ignored.
    /// - The loan is in `self.checked_loans` and in this set: Then the loan should be ignored.
    ignored_loans: ThinBitSet<BorrowIndex>,

    /// All regions that flows forward.
    forward_regions: ThinBitSet<RegionVid>,
    /// All regions that flows backward.
    backward_regions: ThinBitSet<RegionVid>,

    /// All outlives constraints.
    constraints: Constraints<'a, 'tcx>,

    /// A mapping from loans to sets of points where the loans are in scope.
    loan_scopes: IndexVec<BorrowIndex, Option<ThinBitSet<PointIndex>>>,

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

    /// Information of when loan's are killed.
    kills: IndexVec<BorrowIndex, IndexVec<PoloniusBlock, OnceCell<KillAtBlock>>>,

    tcx: TyCtxt<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    body: &'a Body<'tcx>,
    location_map: &'a DenseLocationMap,
    borrow_set: &'a BorrowSet<'tcx>,
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
    fn introduction_block(borrow: &BorrowData<'_>) -> Self {
        Self::from_basic_block(borrow.reserve_location.block)
    }

    /// Get the "before introduction block". I.E the block consisting of statements up to and
    /// including the loan's reserve location.
    #[inline]
    fn before_introduction_block(body: &Body<'_>) -> Self {
        Self::from_usize(body.basic_blocks.len())
    }

    /// Get the correct block from a loan and a location.
    #[inline]
    fn from_location(body: &Body<'_>, borrow: &BorrowData<'_>, location: Location) -> Self {
        if location.block == borrow.reserve_location.block
            && location.statement_index <= borrow.reserve_location.statement_index
        {
            Self::before_introduction_block(body)
        } else {
            Self::from_basic_block(location.block)
        }
    }

    /// Returns the number of polonius blocks. THat is, the number of blocks + 1.
    #[inline]
    fn num_blocks(body: &Body<'_>) -> usize {
        body.basic_blocks.len() + 1
    }

    /// Get the [`BasicBlock`] containing this [`PoloniusBlock``].
    #[inline]
    fn basic_block(self, body: &Body<'_>, borrow: &BorrowData<'_>) -> BasicBlock {
        if self.as_usize() == body.basic_blocks.len() {
            borrow.reserve_location.block
        } else {
            BasicBlock::from_u32(self.as_u32())
        }
    }

    /// Check if this is the "introduction block". I.E the block immediately after the loan has been
    /// introduced.
    #[inline]
    fn is_introduction_block(self, borrow: &BorrowData<'_>) -> bool {
        self.as_u32() == borrow.reserve_location.block.as_u32()
    }

    /// Check if this is the "before introduction block". I.E the block containing statements up to
    /// and including the loan's reserve location.
    #[inline]
    fn is_before_introduction_block(self, body: &Body<'_>) -> bool {
        self.as_usize() == body.basic_blocks.len()
    }

    /// Get the index of the first statement in this block. This will be 0 except for the
    /// introduction block.
    #[inline]
    fn first_index(self, borrow: &BorrowData<'_>) -> usize {
        if self.is_introduction_block(borrow) {
            borrow.reserve_location.statement_index + 1
        } else {
            0
        }
    }

    /// Get the last statement index for this block. For all blocks except the "before introduction
    /// block", this will point to a terminator, not a statement.
    #[inline]
    fn last_index(self, body: &Body<'_>, borrow: &BorrowData<'_>) -> usize {
        if !self.is_before_introduction_block(body) {
            body.basic_blocks[self.basic_block(body, borrow)].statements.len()
        } else {
            borrow.reserve_location.statement_index
        }
    }

    /// Iterate over the successor blocks to this block.
    ///
    /// Note that this is same as [`Terminator::successors`] except for the "before introduction
    /// block" where it is the "introduction block".
    #[inline]
    fn successors(
        self,
        body: &Body<'_>,
        borrow: &BorrowData<'_>,
    ) -> impl DoubleEndedIterator<Item = PoloniusBlock> {
        if !self.is_before_introduction_block(body) {
            Either::Left(body[self.basic_block(body, borrow)].terminator().successors().map(|bb| {
                if bb == borrow.reserve_location.block {
                    Self::before_introduction_block(body)
                } else {
                    Self::from_basic_block(bb)
                }
            }))
        } else {
            Either::Right([Self::introduction_block(borrow)].into_iter())
        }
    }
}

struct LoanRegionNode {
    associated_regions: ThinBitSet<RegionVid>,
    added_regions: Option<ThinBitSet<RegionVid>>,
    /// Whether this location is reachable by forward edges from the loan's introduction point in
    /// the localized constraint graph.
    reachable_by_loan: bool,
    /// Whether the loan is in scope.
    ///
    /// It not possible for a loan to be in scope unless `reachable_by_loan` is true.
    in_scope: bool,
    /// Whether this node has been added to the stack for processing.
    added_to_stack: bool,
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

impl<'a, 'tcx> PoloniusOutOfScopePrecomputer<'a, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        regioncx: &'a RegionInferenceContext<'tcx>,
        body: &'a Body<'tcx>,
        location_map: &'a DenseLocationMap,
        borrow_set: &'a BorrowSet<'tcx>,
        live_region_variances: &'a BTreeMap<RegionVid, ConstraintDirection>,
    ) -> Self {
        my_println!(
            "Universal: {:?}",
            regioncx.universal_regions().universal_regions_iter().collect::<Vec<RegionVid>>()
        );

        // FIXME: Only for debugging.
        my_print!("    Live regions:");
        for region in regioncx.liveness_constraints().regions() {
            my_println!(
                "  {:?}: {:?}",
                region,
                regioncx
                    .liveness_constraints()
                    .points()
                    .row(region)
                    .iter()
                    .flat_map(|pts| pts.iter())
                    .map(|x| regioncx.liveness_constraints().location_from_point(x))
                    .collect::<Vec<_>>()
            );
        }

        let mut forward_regions = new_empty_region_set(regioncx);
        let mut backward_regions = forward_regions.clone();
        for region in (0..num_regions(regioncx)).map(RegionVid::from_usize) {
            match live_region_variances.get(&region) {
                Some(ConstraintDirection::Forward) => {
                    forward_regions.insert(region);
                }
                Some(ConstraintDirection::Backward) => {
                    backward_regions.insert(region);
                }
                Some(ConstraintDirection::Bidirectional) | None => {
                    forward_regions.insert(region);
                    if !regioncx.universal_regions().is_universal_region(region) {
                        backward_regions.insert(region);
                    }
                }
            }
        }
        my_println!("Forward regions: {:?}", forward_regions);
        my_println!("Backward regions: {:?}", backward_regions);
        my_println!("Live region variances: {:?}", live_region_variances);

        let mut constraints = Constraints::new(tcx, regioncx, body, location_map);
        for constraint in regioncx.outlives_constraints() {
            constraints.add_constraint(&constraint);
        }

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
            checked_loans: ThinBitSet::new_empty(borrow_set.len()),
            ignored_loans: ThinBitSet::new_empty(borrow_set.len()),
            loan_scopes: IndexVec::from_elem_n(None, borrow_set.len()),
            constraints,
            forward_regions,
            backward_regions,
            transitive_predecessors,
            adjacent_predecessors,
            kills: IndexVec::from_elem_n(IndexVec::new(), borrow_set.len()),
            tcx,
            regioncx,
            body,
            location_map,
            borrow_set,
        }
    }

    /// Check if a loan is in scope at a location.
    pub(crate) fn loan_in_scope_at(&mut self, borrow_idx: BorrowIndex, location: Location) -> bool {
        let borrow = &self.borrow_set[borrow_idx];

        // Check if this borrow is ignored.
        if !self.checked_loans.insert(borrow_idx) {
            if self.ignored_loans.contains(borrow_idx) {
                return false;
            }
        } else if borrow.borrowed_place().ignore_borrow(
            self.tcx,
            self.body,
            &self.borrow_set.locals_state_at_exit,
        ) {
            self.ignored_loans.insert(borrow_idx);
            return false;
        }

        // Check if this location can never be reached by the borrow.
        if !self.is_predecessor(borrow.reserve_location(), location) {
            return false;
        }

        let point = self.location_map.point_from_location(location);
        if let Some(in_scope_points) = &self.loan_scopes[borrow_idx] {
            return in_scope_points.contains(point);
        }

        // Check if the loan is killed anywhere between its reserve location and `location`.
        let Some(_live_paths) = self.live_paths(borrow_idx, location) else {
            return false;
        };

        let in_scope_points = self.compute_loan_out_of_scope(borrow_idx);
        self.loan_scopes[borrow_idx].insert(in_scope_points).contains(point)
    }

    fn compute_loan_out_of_scope(&mut self, loan_idx: BorrowIndex) -> ThinBitSet<PointIndex> {
        my_println!("- Loan {:?}", loan_idx);
        let loan_data = &self.borrow_set[loan_idx];
        let mut in_scope_points = ThinBitSet::new_empty(self.location_map.num_points());

        // Put the loan's initial region in a set.
        let mut initial_region_set = new_empty_region_set(self.regioncx);
        initial_region_set.insert(loan_data.region);

        let mut nodes = FxHashMap::default();
        let mut stack = Vec::new();
        nodes.insert(
            loan_data.reserve_location,
            LoanRegionNode {
                associated_regions: new_empty_region_set(self.regioncx),
                added_regions: Some(initial_region_set),
                reachable_by_loan: true,
                in_scope: false,
                added_to_stack: true,
            },
        );
        stack.push(loan_data.reserve_location);

        while let Some(location) = stack.pop() {
            let point = self.location_map.point_from_location(location);
            let block_data = &self.body[location.block];

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
            } = nodes.get_mut(&location).unwrap();
            let reachable_by_loan = *reachable_by_loan; // Make copy.
            let in_scope = *in_scope; // Make copy.

            debug_assert!(*added_to_stack);
            *added_to_stack = false;

            let time_travelling_regions = if let Some(mut added_regions) = added_regions.take() {
                debug_assert!(!added_regions.is_empty(), "added_regions should never be empty.");
                debug_assert!(
                    added_regions.iter().all(|r| !associated_regions.contains(r)),
                    "added_regions and associated_regions should be disjunct."
                );

                // Add constraints.
                let time_travelling_regions =
                    self.constraints.add_dependent_regions_at_point(point, &mut added_regions);
                if let Some(tf) = &time_travelling_regions.to_next_loc {
                    my_println!("    Forward time travellers: {:?}", tf);
                }
                if let Some(tf) = &time_travelling_regions.to_prev_stmt {
                    my_println!("    Backward time travellers: {:?}", tf);
                }
                if let Some(x) = &time_travelling_regions.to_preceeding_blocks {
                    my_println!("    To preceeding blocks: {:?}", x);
                }
                if let Some(x) = &time_travelling_regions.to_succeeding_blocks {
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
                    self.remove_dead_regions(location, &mut associated_regions);
                    if associated_regions.is_empty() {
                        my_println!("  Loan killed.");
                        continue;
                    }
                } else {
                    debug_assert!(!in_scope, "If it's not reachable then it's not in scope.");
                    continue;
                }

                None
            };

            // FIXME: This is just a hack.
            {
                let mut associated_regions = associated_regions.clone();
                self.remove_dead_regions(location, &mut associated_regions);
                if in_scope && !associated_regions.is_empty() {
                    in_scope_points.insert(point);
                    my_println!("    In scope at {location:?}");
                }
            }

            // Update in_scope.
            let successor_in_scope = location == loan_data.reserve_location
                || in_scope && self.successor_in_scope(loan_idx, location);

            // Check if the loan is killed.
            let is_killed = !self.successor_in_scope(loan_idx, location)
                && self.is_predecessor(loan_data.reserve_location, location);

            // Make copies of `associated_regions` as that borrow will be killed soon.
            let mut forward_regions = associated_regions.clone();
            let mut backward_regions = associated_regions.clone();

            // Check for forward regions.
            forward_regions.intersect(&self.forward_regions);
            if location.statement_index < block_data.statements.len() {
                let successor_location = location.successor_within_block();
                let successor_node =
                    nodes.entry(successor_location).or_insert_with(|| LoanRegionNode {
                        associated_regions: new_empty_region_set(self.regioncx),
                        added_regions: None,
                        reachable_by_loan: false,
                        in_scope: false,
                        added_to_stack: false,
                    });
                if !is_killed {
                    self.remove_dead_regions(location, &mut forward_regions);
                    self.remove_dead_regions(successor_location, &mut forward_regions);
                    if let Some(tr) = &time_travelling_regions {
                        if let Some(time_travellers) = &tr.to_next_loc {
                            forward_regions.union(time_travellers);
                        }
                    }
                    forward_regions.subtract(&successor_node.associated_regions);
                } else {
                    forward_regions.clear();
                }
                let mut successor_has_changed = false;
                if !forward_regions.is_empty() {
                    my_println!("    Found forward regions: {:?}", forward_regions);
                    if let Some(added_regions) = successor_node.added_regions.as_mut() {
                        added_regions.union(&forward_regions);
                    } else {
                        successor_node.added_regions = Some(forward_regions);
                    }
                    successor_has_changed = true;
                }
                if reachable_by_loan && !successor_node.reachable_by_loan {
                    successor_node.reachable_by_loan = reachable_by_loan;
                    successor_has_changed = true;
                }
                if successor_in_scope && !successor_node.in_scope {
                    successor_node.in_scope = successor_in_scope;
                    successor_has_changed = true;
                }
                if successor_has_changed && !successor_node.added_to_stack {
                    stack.push(successor_location);
                    successor_node.added_to_stack = true;
                }
            } else {
                for successor_block in block_data.terminator().successors() {
                    let mut forward_regions = forward_regions.clone();
                    let successor_location =
                        Location { block: successor_block, statement_index: 0 };
                    let successor_node =
                        nodes.entry(successor_location).or_insert_with(|| LoanRegionNode {
                            associated_regions: new_empty_region_set(self.regioncx),
                            added_regions: None,
                            reachable_by_loan: false,
                            in_scope: false,
                            added_to_stack: false,
                        });
                    if !is_killed {
                        self.remove_dead_regions(location, &mut forward_regions);
                        self.remove_dead_regions(successor_location, &mut forward_regions);
                        if let Some(tr) = &time_travelling_regions {
                            if let Some(time_travellers) = tr
                                .to_succeeding_blocks
                                .as_ref()
                                .and_then(|x| x.row(successor_block))
                            {
                                forward_regions.union(time_travellers);
                            }
                        }
                        forward_regions.subtract(&successor_node.associated_regions);
                    } else {
                        forward_regions.clear();
                    }

                    let mut successor_has_changed = false;
                    if !forward_regions.is_empty() {
                        my_println!(
                            "    Found forward regions to {:?}: {:?}",
                            successor_location,
                            forward_regions
                        );
                        if let Some(added_regions) = successor_node.added_regions.as_mut() {
                            added_regions.union(&forward_regions);
                        } else {
                            successor_node.added_regions = Some(forward_regions);
                        }
                        successor_has_changed = true;
                    }
                    if reachable_by_loan && !successor_node.reachable_by_loan {
                        successor_node.reachable_by_loan = reachable_by_loan;
                        successor_has_changed = true;
                    }
                    if successor_in_scope && !successor_node.in_scope {
                        successor_node.in_scope = successor_in_scope;
                        successor_has_changed = true;
                    }
                    if successor_has_changed && !successor_node.added_to_stack {
                        stack.push(successor_location);
                        successor_node.added_to_stack = true;
                    }
                }
            }

            // Check for backward regions.
            backward_regions.intersect(&self.backward_regions);
            if location.statement_index > 0 {
                let predecessor_location = location.predecessor_within_block();
                let predecessor_node =
                    nodes.entry(predecessor_location).or_insert_with(|| LoanRegionNode {
                        associated_regions: new_empty_region_set(self.regioncx),
                        added_regions: None,
                        reachable_by_loan: false,
                        in_scope: false,
                        added_to_stack: false,
                    });
                // To comply with previous Polonius, this if condition was:
                // `!is_killed || !location.is_predecessor_of(predecessor_location)`
                // But it doesn't seem to be needed to pass the tests.
                if !is_killed {
                    self.remove_dead_regions(location, &mut backward_regions);
                    self.remove_dead_regions(predecessor_location, &mut backward_regions);
                    if let Some(tr) = &time_travelling_regions {
                        if let Some(time_travellers) = &tr.to_prev_stmt {
                            backward_regions.union(time_travellers);
                        }
                    }
                    backward_regions.subtract(&predecessor_node.associated_regions);
                } else {
                    backward_regions.clear();
                }

                if !backward_regions.is_empty() {
                    my_println!("    Found backward regions: {:?}", backward_regions);
                    if let Some(added_regions) = predecessor_node.added_regions.as_mut() {
                        added_regions.union(&backward_regions);
                    } else {
                        predecessor_node.added_regions = Some(backward_regions);
                    }
                    if !predecessor_node.added_to_stack {
                        stack.push(predecessor_location);
                        predecessor_node.added_to_stack = true;
                    }
                }
            } else {
                for &predecessor_block in &self.body.basic_blocks.predecessors()[location.block] {
                    let mut backward_regions = backward_regions.clone();
                    let predecessor_location = Location {
                        block: predecessor_block,
                        statement_index: self.body[predecessor_block].statements.len(),
                    };
                    let predecessor_node =
                        nodes.entry(predecessor_location).or_insert_with(|| LoanRegionNode {
                            associated_regions: new_empty_region_set(self.regioncx),
                            added_regions: None,
                            reachable_by_loan: false,
                            in_scope: false,
                            added_to_stack: false,
                        });
                    if !is_killed {
                        self.remove_dead_regions(location, &mut backward_regions);
                        self.remove_dead_regions(predecessor_location, &mut backward_regions);
                        if let Some(tr) = &time_travelling_regions {
                            if let Some(time_travellers) = tr
                                .to_preceeding_blocks
                                .as_ref()
                                .and_then(|x| x.row(predecessor_block))
                            {
                                backward_regions.union(time_travellers);
                            }
                        }
                        backward_regions.subtract(&predecessor_node.associated_regions);
                    } else {
                        backward_regions.clear();
                    }

                    if !backward_regions.is_empty() {
                        my_println!(
                            "    Found backward regions to {:?}: {:?}",
                            predecessor_location,
                            backward_regions
                        );
                        if let Some(added_regions) = predecessor_node.added_regions.as_mut() {
                            added_regions.union(&backward_regions);
                        } else {
                            predecessor_node.added_regions = Some(backward_regions);
                        }
                        if !predecessor_node.added_to_stack {
                            stack.push(predecessor_location);
                            predecessor_node.added_to_stack = true;
                        }
                    }
                }
            }
        }

        in_scope_points
    }

    /// Remove dead regions from the set of associated regions.
    fn remove_dead_regions(&self, location: Location, region_set: &mut ThinBitSet<RegionVid>) {
        for region in region_set.clone().iter() {
            if !self.regioncx.liveness_constraints().is_live_at(region, location) {
                region_set.remove(region);
            }
        }
    }

    /// Return the `in_scope` value for the successor location(s).
    fn successor_in_scope(&self, borrow_idx: BorrowIndex, location: Location) -> bool {
        if let Some(stmt) = self.body[location.block].statements.get(location.statement_index) {
            !self.kill_at_stmt(borrow_idx, stmt)
        } else {
            !self.kill_at_terminator(borrow_idx, &self.body[location.block].terminator())
        }
    }

    /// Calculate when/if a loan goes out of scope for a set of statements in a block.
    #[inline]
    fn kill_at_block(
        &self,
        borrow_idx: BorrowIndex,
        block_idx: BasicBlock,
        statements: impl IntoIterator<Item = usize>,
    ) -> KillAtBlock {
        let block = &self.body[block_idx];
        if let Some(statement_index) = statements.into_iter().find(|&statement_idx| {
            if let Some(stmt) = block.statements.get(statement_idx) {
                self.kill_at_stmt(borrow_idx, stmt)
            } else {
                self.kill_at_terminator(borrow_idx, &block.terminator())
            }
        }) {
            Killed { statement_index }
        } else {
            NotKilled
        }
    }

    /// Given that the borrow was in scope on entry to this statement, check if it goes out of scope
    /// till the next location.
    #[inline]
    fn kill_at_stmt(&self, borrow_idx: BorrowIndex, stmt: &Statement<'tcx>) -> bool {
        match &stmt.kind {
            mir::StatementKind::Assign(box (lhs, _rhs)) => self.kill_on_place(borrow_idx, *lhs),
            mir::StatementKind::StorageDead(local) => {
                self.borrow_set.local_map.get(local).is_some_and(|bs| bs.contains(&borrow_idx))
            }
            _ => false,
        }
    }

    /// Given that the borrow was in scope on entry to this terminator, check if it goes out of scope
    /// till the succeeding blocks.
    #[inline]
    fn kill_at_terminator(&self, borrow_idx: BorrowIndex, terminator: &Terminator<'tcx>) -> bool {
        match &terminator.kind {
            // A `Call` terminator's return value can be a local which has borrows, so we need to record
            // those as killed as well.
            mir::TerminatorKind::Call { destination, .. } => {
                self.kill_on_place(borrow_idx, *destination)
            }
            mir::TerminatorKind::InlineAsm { operands, .. } => operands.iter().any(|op| {
                if let mir::InlineAsmOperand::Out { place: Some(place), .. }
                | mir::InlineAsmOperand::InOut { out_place: Some(place), .. } = op
                {
                    self.kill_on_place(borrow_idx, *place)
                } else {
                    false
                }
            }),
            _ => false,
        }
    }

    #[inline]
    fn kill_on_place(&self, borrow_idx: BorrowIndex, place: Place<'tcx>) -> bool {
        self.borrow_set.local_map.get(&place.local).is_some_and(|bs| bs.contains(&borrow_idx))
            && if place.projection.is_empty() {
                !self.body.local_decls[place.local].is_ref_to_static()
            } else {
                places_conflict(
                    self.tcx,
                    self.body,
                    self.borrow_set[borrow_idx].borrowed_place,
                    place,
                    PlaceConflictBias::NoOverlap,
                )
            }
    }

    /// Returns `true` iff `a` is earlier in the control flow graph than `b`.
    #[inline]
    fn is_predecessor(&self, a: Location, b: Location) -> bool {
        a.block == b.block && a.statement_index < b.statement_index
            || self.transitive_predecessors[b.block].contains(a.block)
    }

    fn live_paths(
        &mut self,
        borrow_idx: BorrowIndex,
        destination: Location,
    ) -> Option<ThinBitSet<PoloniusBlock>> {
        let borrow = &self.borrow_set[borrow_idx];
        // `destination_block` is the `PoloniusBlock` for `destination`.
        let destination_block = PoloniusBlock::from_location(self.body, borrow, destination);

        // We begin by checking the relevant statements in `destination_block`.
        // FIXME: Check in `self.kills` first.
        if let Killed { .. } = self.kill_at_block(
            borrow_idx,
            destination.block,
            destination_block.first_index(borrow)..destination.statement_index,
        ) {
            return None;
        }

        if destination_block.is_introduction_block(borrow) {
            // We are finished.
            return Some(ThinBitSet::new_empty(PoloniusBlock::num_blocks(self.body)));
        }

        // Traverse all blocks between `reserve_location` and `destination` in the CFG and check for
        // kills. If there is no live path from `reserve_location` to `destination`, we no for sure
        // that the loan is dead at `destination`.

        // Keep track of all visited `PoloniusBlock`s.
        let mut visited = ThinBitSet::new_empty(PoloniusBlock::num_blocks(self.body));

        // The stack contains `(block, path)` pairs, where `block` is a `PoloniusBlock` and `path is
        // a set of `PoloniusBlock`s making a path from `reserve_location` to `destination_block`.
        // In this way we can record the live paths.
        let introduction_block = PoloniusBlock::introduction_block(borrow);
        let mut stack: SmallVec<[(PoloniusBlock, ThinBitSet<PoloniusBlock>); 4]> = smallvec![(
            introduction_block,
            ThinBitSet::new_empty(PoloniusBlock::num_blocks(self.body))
        )];
        visited.insert(introduction_block);

        let mut valid_paths = None;

        while let Some((block, path)) = stack.pop() {
            let basic_block = block.basic_block(self.body, borrow);

            // Check if the loan is killed in this block.
            self.kills[borrow_idx].ensure_contains_elem(block, OnceCell::new);
            if let Killed { .. } = self.kills[borrow_idx][block].get_or_init(|| {
                self.kill_at_block(
                    borrow_idx,
                    basic_block,
                    block.first_index(borrow)..=block.last_index(self.body, borrow),
                )
            }) {
                continue;
            }

            // Loop through all successors to `block` and follow those that are predecessors to
            // `destination.block`.
            for successor in block.successors(self.body, borrow) {
                let successor_bb = successor.basic_block(self.body, borrow);

                if successor == destination_block {
                    // We have reached the destination so let's save this path.
                    valid_paths
                        .get_or_insert_with(|| {
                            ThinBitSet::<PoloniusBlock>::new_empty(PoloniusBlock::num_blocks(
                                self.body,
                            ))
                        })
                        .union(&path);

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
                if !self.transitive_predecessors[destination.block].contains(successor_bb)
                    || destination_block.is_introduction_block(borrow)
                        && successor.is_before_introduction_block(self.body)
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
