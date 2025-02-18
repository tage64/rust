use std::assert_matches::assert_matches;
use std::collections::{BTreeMap, BTreeSet};
use std::mem;
use std::sync::LazyLock;

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_index::bit_set::{DenseBitSet, SparseBitMatrix};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::{
    BasicBlock, Body, Location, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::{TyCtxt, TypeVisitable};
use rustc_mir_dataflow::points::DenseLocationMap;

use super::ConstraintDirection;
use super::loan_liveness::collect_kills;
use crate::constraints::OutlivesConstraint;
use crate::type_check::Locations;
use crate::{BorrowIndex, BorrowSet, RegionInferenceContext, RegionVid};

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

pub(super) fn compute_loans_out_of_scope<'tcx>(
    tcx: TyCtxt<'tcx>,
    regioncx: &mut RegionInferenceContext<'tcx>,
    body: &Body<'tcx>,
    location_map: &DenseLocationMap,
    borrow_set: &BorrowSet<'tcx>,
    live_region_variances: &BTreeMap<RegionVid, ConstraintDirection>,
) -> FxIndexMap<Location, Vec<BorrowIndex>> {
    let mut polonius = PoloniusOutOfScopePrecomputer::new(
        tcx,
        regioncx,
        body,
        location_map,
        borrow_set,
        live_region_variances,
    );
    for borrow_idx in borrow_set.indices() {
        polonius.compute_loan_out_of_scope(borrow_idx);
    }

    let mut loans_out_of_scope_at_location = FxIndexMap::<_, Vec<_>>::default();
    for (loan, deaths) in polonius.loans_out_of_scope.into_iter_enumerated() {
        for (polonius_block, statement_indices) in deaths {
            let block = match polonius_block {
                PoloniusBlock::Normal(b) => b,
                PoloniusBlock::BeforeReserveLocation => borrow_set[loan].reserve_location.block,
            };
            let Some(statement_index) = statement_indices.first().copied() else {
                continue;
            };

            loans_out_of_scope_at_location
                .entry(Location { block, statement_index })
                .or_default()
                .push(loan);
        }
    }
    loans_out_of_scope_at_location
}

struct PoloniusOutOfScopePrecomputer<'a, 'tcx> {
    /// TODO: A map of all loan kills by their location. This should maybe be reworked.
    kills: BTreeMap<Location, BTreeSet<BorrowIndex>>,
    /// All regions that flows forward.
    forward_regions: DenseBitSet<RegionVid>,
    /// All regions that flows backward.
    backward_regions: DenseBitSet<RegionVid>,

    /// A mapping from locations in the CFG to a set of loans that go out of scope. This will be the
    /// final result of the computation.
    loans_out_of_scope: IndexVec<BorrowIndex, FxIndexMap<PoloniusBlock, BTreeSet<usize>>>,

    tcx: TyCtxt<'tcx>,
    regioncx: &'a mut RegionInferenceContext<'tcx>,
    body: &'a Body<'tcx>,
    #[allow(dead_code)] // TODO: I keep this until I know I don't need it.
    location_map: &'a DenseLocationMap,
    borrow_set: &'a BorrowSet<'tcx>,
}

#[derive(Default)]
struct TimeTravellingRegions {
    to_prev_stmt: Option<DenseBitSet<RegionVid>>,
    to_proceeding_blocks: Option<SparseBitMatrix<BasicBlock, RegionVid>>,
    to_next_loc: Option<DenseBitSet<RegionVid>>,
    to_succeeding_blocks: Option<SparseBitMatrix<BasicBlock, RegionVid>>,
}

#[derive(Debug, Copy, Clone)]
enum TimeTravelDirection {
    Backwards,
    Forwards,
}

impl TimeTravellingRegions {
    fn add_within_block(
        &mut self,
        regioncx: &RegionInferenceContext<'_>,
        region: RegionVid,
        direction: TimeTravelDirection,
    ) {
        match direction {
            TimeTravelDirection::Forwards => {
                self.to_next_loc
                    .get_or_insert_with(|| new_empty_region_set(regioncx))
                    .insert(region);
            }
            TimeTravelDirection::Backwards => {
                self.to_prev_stmt
                    .get_or_insert_with(|| new_empty_region_set(regioncx))
                    .insert(region);
            }
        }
    }

    fn add_to_proceeding_block(
        &mut self,
        regioncx: &RegionInferenceContext<'_>,
        region: RegionVid,
        proceeding_block: BasicBlock,
    ) {
        self.to_proceeding_blocks
            .get_or_insert_with(|| new_region_matrix(regioncx))
            .insert(proceeding_block, region);
    }

    fn add_to_succeeding_block(
        &mut self,
        regioncx: &RegionInferenceContext<'_>,
        region: RegionVid,
        succeeding_block: BasicBlock,
    ) {
        self.to_succeeding_blocks
            .get_or_insert_with(|| new_region_matrix(regioncx))
            .insert(succeeding_block, region);
    }
}

/// A `PoloniusBlock` is a `BasicBlock` which distinguishes between before and after the reserve
/// location of a particular loan.
///
/// The problem is that we want to record at most one location per block where a loan goes out of
/// scope. But a loan might go out of scope twice in the block where it is created, either before or
/// after the reserve location. So we use a special variant to denote the case when the loan goes
/// out of scope at a earlier statement than the reserve location but in the same block.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum PoloniusBlock {
    /// This denotes all statements up to and including the reserve location in the block where the
    /// loan is reserved.
    BeforeReserveLocation,
    /// The same as a normal `Basicblock` excluding all statements before and including the reserve
    /// location in the reserve block.
    Normal(BasicBlock),
}

struct LoanRegionNode {
    associated_regions: DenseBitSet<RegionVid>,
    added_regions: Option<DenseBitSet<RegionVid>>,
    reachable_by_loan: bool,
    added_to_stack: bool,
}

impl<'a, 'tcx> PoloniusOutOfScopePrecomputer<'a, 'tcx> {
    fn new(
        tcx: TyCtxt<'tcx>,
        regioncx: &'a mut RegionInferenceContext<'tcx>,
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
        let kills = collect_kills(body, tcx, borrow_set);

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

        Self {
            loans_out_of_scope: IndexVec::from_fn_n(|_| Default::default(), borrow_set.len()),
            kills,
            forward_regions,
            backward_regions,
            tcx,
            regioncx,
            body,
            location_map,
            borrow_set,
        }
    }

    fn compute_loan_out_of_scope(&mut self, loan_idx: BorrowIndex) {
        my_println!("- Loan {:?}", loan_idx);
        let loan_data = &self.borrow_set[loan_idx];

        // Put the loan's initial region in a set.
        let mut initial_region_set = new_empty_region_set(self.regioncx);
        initial_region_set.insert(loan_data.region);

        let mut nodes = FxHashMap::default();
        let mut stack = Vec::new();
        nodes.insert(loan_data.reserve_location, LoanRegionNode {
            associated_regions: new_empty_region_set(self.regioncx),
            added_regions: Some(initial_region_set),
            reachable_by_loan: true,
            added_to_stack: true,
        });
        stack.push(loan_data.reserve_location);

        while let Some(location) = stack.pop() {
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
                added_to_stack,
            } = nodes.get_mut(&location).unwrap();
            let reachable_by_loan = *reachable_by_loan; // Make copy.

            debug_assert!(*added_to_stack);
            *added_to_stack = false;

            let Some(mut added_regions) = added_regions.take() else {
                my_println!("Nothing new here.");
                if reachable_by_loan {
                    if associated_regions.is_empty() {
                        my_println!("  Loan killed.");
                        self.add_kill(loan_idx, location);
                    } else {
                        // Propagate reachability to succeeding nodes.
                        // TODO: Check if this is really the best approach.
                        if location.statement_index < block_data.statements.len() {
                            let successor_location = location.successor_within_block();
                            let LoanRegionNode {
                                reachable_by_loan: succ_reachable,
                                added_to_stack,
                                ..
                            } = nodes.get_mut(&location.successor_within_block()).unwrap();
                            if !*succ_reachable {
                                my_println!(
                                    "    Propagating reachability to {successor_location:?}."
                                );
                                *succ_reachable = true;
                                if !*added_to_stack {
                                    stack.push(successor_location);
                                    *added_to_stack = true;
                                }
                            }
                        } else {
                            for successor_block in block_data.terminator().successors() {
                                let successor_location =
                                    Location { block: successor_block, statement_index: 0 };
                                let LoanRegionNode {
                                    reachable_by_loan: succ_reachable,
                                    added_to_stack,
                                    ..
                                } = nodes.get_mut(&successor_location).unwrap();
                                if !*succ_reachable {
                                    my_println!(
                                        "    Propagating reachability to {successor_location:?}."
                                    );
                                    *succ_reachable = true;
                                    if !*added_to_stack {
                                        stack.push(successor_location);
                                        *added_to_stack = true;
                                    }
                                }
                            }
                        }
                    }
                }
                continue;
            };

            debug_assert!(!added_regions.is_empty(), "added_regions should never be empty.");
            debug_assert!(
                added_regions.iter().all(|r| !associated_regions.contains(r)),
                "added_regions and associated_regions should be disjunct."
            );

            // Add constraints.
            let time_travelling_regions =
                self.add_dependent_regions_at_location(location, &mut added_regions);
            if let Some(tf) = &time_travelling_regions.to_next_loc {
                my_println!("    Forward time travellers: {:?}", tf);
            }
            if let Some(tf) = &time_travelling_regions.to_prev_stmt {
                my_println!("    Backward time travellers: {:?}", tf);
            }
            if let Some(x) = &time_travelling_regions.to_proceeding_blocks {
                my_println!("    To proceeding blocks: {:?}", x);
            }
            if let Some(x) = &time_travelling_regions.to_succeeding_blocks {
                my_println!("    To succeeding blocks: {:?}", x);
            }

            // TODO: Look if this is necessary
            if associated_regions.is_empty() {
                debug_assert!(!added_regions.is_empty());
                self.remove_kill(loan_idx, location);
            }

            // Incorporate the added regions into `associated_regions`.
            associated_regions.union(&added_regions);
            my_println!("    Regions: {:?}", associated_regions);

            // Check if the loan is killed.
            let is_killed = self.kills.get(&location).is_some_and(|x| x.contains(&loan_idx));

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
                        reachable_by_loan,
                        added_to_stack: false,
                    });
                if !is_killed {
                    if let Some(time_travellers) = &time_travelling_regions.to_next_loc {
                        forward_regions.union(time_travellers);
                    }
                    forward_regions.subtract(&successor_node.associated_regions);
                    self.remove_dead_regions(successor_location, &mut forward_regions);
                } else {
                    forward_regions.clear();
                }
                if !forward_regions.is_empty() {
                    my_println!("    Found forward regions: {:?}", forward_regions);
                    if let Some(added_regions) = successor_node.added_regions.as_mut() {
                        added_regions.union(&forward_regions);
                    } else {
                        successor_node.added_regions = Some(forward_regions);
                    }
                }
                successor_node.reachable_by_loan |= reachable_by_loan;
                if !successor_node.added_to_stack {
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
                            reachable_by_loan,
                            added_to_stack: false,
                        });
                    if !is_killed {
                        if let Some(time_travellers) = time_travelling_regions
                            .to_succeeding_blocks
                            .as_ref()
                            .and_then(|x| x.row(successor_block))
                        {
                            forward_regions.union(time_travellers);
                        }
                        forward_regions.subtract(&successor_node.associated_regions);
                        self.remove_dead_regions(successor_location, &mut forward_regions);
                    } else {
                        forward_regions.clear();
                    }

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
                    }
                    successor_node.reachable_by_loan |= reachable_by_loan;
                    if !successor_node.added_to_stack {
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
                        added_to_stack: false,
                    });
                if !(is_killed && location.is_predecessor_of(predecessor_location, self.body)) {
                    if let Some(time_travellers) = &time_travelling_regions.to_prev_stmt {
                        backward_regions.union(time_travellers);
                    }
                    backward_regions.subtract(&predecessor_node.associated_regions);
                    self.remove_dead_regions(predecessor_location, &mut backward_regions);
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
                }
                if !predecessor_node.added_to_stack {
                    stack.push(predecessor_location);
                    predecessor_node.added_to_stack = true;
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
                            added_to_stack: false,
                        });
                    if !(is_killed && location.is_predecessor_of(predecessor_location, self.body)) {
                        if let Some(time_travellers) = time_travelling_regions
                            .to_proceeding_blocks
                            .as_ref()
                            .and_then(|x| x.row(predecessor_block))
                        {
                            backward_regions.union(time_travellers);
                        }
                        backward_regions.subtract(&predecessor_node.associated_regions);
                        self.remove_dead_regions(predecessor_location, &mut backward_regions);
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
                    }
                    if !predecessor_node.added_to_stack {
                        stack.push(predecessor_location);
                        predecessor_node.added_to_stack = true;
                    }
                }
            }
        }
    }

    /// Remove dead regions from the set of associated regions.
    fn remove_dead_regions(&self, location: Location, region_set: &mut DenseBitSet<RegionVid>) {
        for region in region_set.clone().iter() {
            if !self.regioncx.liveness_constraints().is_live_at(region, location) {
                region_set.remove(region);
            }
        }
    }

    // TODO: This algorithm is extremly slow.
    fn add_dependent_regions_at_location(
        &self,
        location: Location,
        associated_regions: &mut DenseBitSet<RegionVid>,
    ) -> TimeTravellingRegions {
        // FIXME: Only for debugging.
        my_println!("    Regions for constraints: {:?}", associated_regions);
        for constraint in self.regioncx.outlives_constraints() {
            if let Locations::Single(l) = constraint.locations {
                if l == location {
                    my_println!("      {:?}: {:?}", constraint.sup, constraint.sub);
                }
            }
        }

        let mut to_check = associated_regions.clone();
        let mut to_check_next_round = new_empty_region_set(self.regioncx);
        let mut time_travelling_regions = TimeTravellingRegions::default();

        while !to_check.is_empty() {
            for constraint in self.regioncx.outlives_constraints() {
                if !to_check.contains(constraint.sup) {
                    continue;
                }

                if self.add_time_traveller(location, &constraint, &mut time_travelling_regions) {
                    // If the region is time travelling we should not add it to
                    // `associated_regions`.
                    continue;
                }

                if let Locations::Single(l) = constraint.locations
                    && l != location
                {
                    continue;
                }

                if associated_regions.insert(constraint.sub) {
                    to_check_next_round.insert(constraint.sub);
                }
            }
            mem::swap(&mut to_check, &mut to_check_next_round);
            to_check_next_round.clear();
        }

        time_travelling_regions
    }

    /// Check if this constraint is travelling in time and if so add it to `time_travellers` and
    /// return true, otherwise return false.
    fn add_time_traveller(
        &self,
        location: Location,
        constraint: &OutlivesConstraint<'tcx>,
        time_travellers: &mut TimeTravellingRegions,
    ) -> bool {
        match constraint.locations {
            Locations::Single(l) if l == location => {
                if let Some(stmt) = self.body[location.block].statements.get(l.statement_index) {
                    match self.time_traveller_at_statement(constraint, stmt) {
                        Some(t @ TimeTravelDirection::Forwards) => {
                            time_travellers.add_within_block(self.regioncx, constraint.sub, t);
                            true
                        }
                        Some(TimeTravelDirection::Backwards) | None => false,
                    }
                } else {
                    debug_assert_eq!(l.statement_index, self.body[l.block].statements.len());
                    let terminator = self.body[l.block].terminator();
                    match self.time_traveller_at_terminator(constraint, terminator) {
                        Some((TimeTravelDirection::Forwards, target_block)) => {
                            time_travellers.add_to_succeeding_block(
                                self.regioncx,
                                constraint.sub,
                                target_block,
                            );
                            true
                        }
                        Some((TimeTravelDirection::Backwards, _)) | None => false,
                    }
                }
            }
            Locations::Single(l) if l.successor_within_block() == location => {
                let stmt = self.body[location.block].statements.get(l.statement_index).unwrap();
                match self.time_traveller_at_statement(constraint, stmt) {
                    Some(t @ TimeTravelDirection::Backwards) => {
                        time_travellers.add_within_block(self.regioncx, constraint.sub, t);
                        true
                    }
                    Some(TimeTravelDirection::Forwards) | None => false,
                }
            }
            Locations::Single(l) => {
                let block_data = &self.body[l.block];
                if l.statement_index == block_data.statements.len()
                    && block_data.terminator().successors().any(|b| b == location.block)
                {
                    let terminator = self.body[l.block].terminator();
                    match self.time_traveller_at_terminator(constraint, terminator) {
                        Some((TimeTravelDirection::Backwards, source_block)) => {
                            time_travellers.add_to_proceeding_block(
                                self.regioncx,
                                constraint.sub,
                                source_block,
                            );
                            true
                        }
                        Some((TimeTravelDirection::Forwards, _)) | None => false,
                    }
                } else {
                    false
                }
            }
            Locations::All(_) => false,
        }
    }

    fn time_traveller_at_statement(
        &self,
        constraint: &OutlivesConstraint<'tcx>,
        statement: &Statement<'tcx>,
    ) -> Option<TimeTravelDirection> {
        match &statement.kind {
            StatementKind::Assign(box (lhs, rhs)) => {
                // TODO: Check this comment:
                // To create localized outlives constraints without midpoints, we rely on the property
                // that no input regions from the RHS of the assignment will flow into themselves: they
                // should not appear in the output regions in the LHS. We believe this to be true by
                // construction of the MIR, via temporaries, and assert it here.
                //
                // We think we don't need midpoints because:
                // - every LHS Place has a unique set of regions that don't appear elsewhere
                // - this implies that for them to be part of the RHS, the same Place must be read and
                //   written
                // - and that should be impossible in MIR
                //
                // When we have a more complete implementation in the future, tested with crater, etc,
                // we can maybe remove this assertion.
                debug_assert!(
                    {
                        let mut lhs_regions = FxHashSet::default();
                        self.tcx.for_each_free_region(lhs, |region| {
                            let region = self.regioncx.universal_regions().to_region_vid(region);
                            lhs_regions.insert(region);
                        });

                        let mut rhs_regions = FxHashSet::default();
                        self.tcx.for_each_free_region(rhs, |region| {
                            let region = self.regioncx.universal_regions().to_region_vid(region);
                            rhs_regions.insert(region);
                        });

                        // The intersection between LHS and RHS regions should be empty.
                        lhs_regions.is_disjoint(&rhs_regions)
                    },
                    "there should be no common regions between the LHS and RHS of an assignment"
                );

                // As mentioned earlier, we should be tracking these better upstream but: we want to
                // relate the types on entry to the type of the place on exit. That is, outlives
                // constraints on the RHS are on entry, and outlives constraints to/from the LHS are on
                // exit (i.e. on entry to the successor location).
                let lhs_ty = self.body.local_decls[lhs.local].ty;
                self.compute_constraint_direction(constraint, &lhs_ty)
            }
            _ => None,
        }
    }

    fn time_traveller_at_terminator(
        &self,
        constraint: &OutlivesConstraint<'tcx>,
        terminator: &Terminator<'tcx>,
    ) -> Option<(TimeTravelDirection, BasicBlock)> {
        // FIXME: check if other terminators need the same handling as `Call`s, in particular
        // Assert/Yield/Drop. A handful of tests are failing with Drop related issues, as well as some
        // coroutine tests, and that may be why.
        match &terminator.kind {
            // FIXME: also handle diverging calls.
            TerminatorKind::Call { destination, target: Some(target_block), .. } => {
                // Calls are similar to assignments, and thus follow the same pattern. If there is a
                // target for the call we also relate what flows into the destination here to entry to
                // that successor.
                let destination_ty = destination.ty(&self.body.local_decls, self.tcx);
                self.compute_constraint_direction(constraint, &destination_ty)
                    .map(|t| (t, *target_block))
            }
            _ => None,
        }
    }

    /// For a given outlives constraint and CFG edge, returns the localized constraint with the
    /// appropriate `from`-`to` direction. This is computed according to whether the constraint flows to
    /// or from a free region in the given `value`, some kind of result for an effectful operation, like
    /// the LHS of an assignment.
    fn compute_constraint_direction(
        &self,
        constraint: &OutlivesConstraint<'tcx>,
        value: &impl TypeVisitable<TyCtxt<'tcx>>,
    ) -> Option<TimeTravelDirection> {
        let mut dir = None;
        self.tcx.for_each_free_region(value, |region| {
            let region = self.regioncx.universal_regions().to_region_vid(region);
            if region == constraint.sub {
                // This constraint flows into the result, its effects start becoming visible on exit.
                assert_matches!(dir, None | Some(TimeTravelDirection::Forwards));
                dir = Some(TimeTravelDirection::Forwards);
            } else if region == constraint.sup {
                // This constraint flows from the result, its effects start becoming visible on exit.
                assert_matches!(dir, None | Some(TimeTravelDirection::Backwards));
                dir = Some(TimeTravelDirection::Backwards);
            }
        });
        dir
    }

    fn add_kill(&mut self, loan_idx: BorrowIndex, location: Location) {
        let reserve_location = self.borrow_set[loan_idx].reserve_location;
        let block = if location.block == reserve_location.block
            && location.statement_index < reserve_location.statement_index
        {
            PoloniusBlock::BeforeReserveLocation
        } else if location == reserve_location {
            // We don't make a kill at the reserve location.
            return;
        } else {
            PoloniusBlock::Normal(location.block)
        };

        self.loans_out_of_scope[loan_idx]
            .entry(block)
            .or_insert_with(BTreeSet::default)
            .insert(location.statement_index);
    }

    fn remove_kill(&mut self, loan_idx: BorrowIndex, location: Location) {
        let reserve_location = self.borrow_set[loan_idx].reserve_location;
        let block = if location.block == reserve_location.block
            && location.statement_index < reserve_location.statement_index
        {
            PoloniusBlock::BeforeReserveLocation
        } else if location == reserve_location {
            // We don't make a kill at the reserve location.
            return;
        } else {
            PoloniusBlock::Normal(location.block)
        };

        if let Some(statement_indices) = self.loans_out_of_scope[loan_idx].get_mut(&block) {
            statement_indices.remove(&location.statement_index);
        }
    }
}

/// Create an empty bit set with capacity for all regions.
fn new_empty_region_set(regioncx: &RegionInferenceContext<'_>) -> DenseBitSet<RegionVid> {
    DenseBitSet::new_empty(regioncx.last_region_vid().map_or(0, |x| x.index() + 1))
}

fn new_region_matrix<R: Idx>(
    regioncx: &RegionInferenceContext<'_>,
) -> SparseBitMatrix<R, RegionVid> {
    SparseBitMatrix::new(num_regions(regioncx))
}

fn num_regions(regioncx: &RegionInferenceContext<'_>) -> usize {
    regioncx.last_region_vid().map_or(0, |x| x.index() + 1)
}

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
