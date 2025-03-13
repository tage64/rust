#![allow(dead_code)]
#![deny(unused_imports)]
mod constraints;
mod loan_invalidations;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use constraints::Constraints;
use loan_invalidations::compute_loan_invalidations;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_index::bit_set::thin_bit_set::{SparseBitMatrix, ThinBitSet};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::{self, BasicBlock, Body, Location, Place, Statement, Terminator};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::points::{DenseLocationMap, PointIndex};

use super::ConstraintDirection;
use super::loan_liveness::collect_kills;
use crate::{
    BorrowIndex, BorrowSet, PlaceConflictBias, PlaceExt, RegionInferenceContext, RegionVid,
    places_conflict,
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

pub(super) fn compute_loans_out_of_scope<'tcx>(
    tcx: TyCtxt<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
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

    let loan_invalidations = compute_loan_invalidations(tcx, body, borrow_set);

    for borrow_idx in borrow_set.indices() {
        let invalidation_locations = &loan_invalidations[borrow_idx];
        if invalidation_locations.is_empty() {
            if !body.local_decls[borrow_set[borrow_idx].borrowed_place.local]
                .is_ref_to_thread_local()
            {
                my_println!("Loan {borrow_idx:?} is never invalidated.");
                continue;
            }
        } else if loan_invalidations[borrow_idx].iter().all(|&invalidation_location| {
            let mut associated_regions = new_empty_region_set(regioncx);
            associated_regions.insert(borrow_set[borrow_idx].region);
            polonius.constraints.add_dependent_regions(&mut associated_regions);
            polonius.remove_dead_regions(invalidation_location, &mut associated_regions);
            my_println!("Invalidated at {invalidation_location:?}");
            if associated_regions.is_empty() {
                //polonius.add_kill(borrow_idx, invalidation_location);
                true
            } else {
                false
            }
        }) {
            my_println!("Loan {borrow_idx:?} is never invalidated.");
            //continue;
        }

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
    my_println!("Loans out of scope at location: {loans_out_of_scope_at_location:?}");
    loans_out_of_scope_at_location
}

pub(crate) struct PoloniusOutOfScopePrecomputer<'a, 'tcx> {
    /// TODO: A map of all loan kills by their location. This should maybe be reworked.
    kills: BTreeMap<Location, BTreeSet<BorrowIndex>>,
    /// All regions that flows forward.
    forward_regions: ThinBitSet<RegionVid>,
    /// All regions that flows backward.
    backward_regions: ThinBitSet<RegionVid>,

    /// All outlives constraints.
    constraints: Constraints<'a, 'tcx>,

    /// A mapping from locations in the CFG to a set of loans that go out of scope. This will be the
    /// final result of the computation.
    loans_out_of_scope: IndexVec<BorrowIndex, FxIndexMap<PoloniusBlock, BTreeSet<usize>>>,
    /// A mapping from loans to sets of points where the loans are in scope.
    loan_scopes: IndexVec<BorrowIndex, Option<ThinBitSet<PointIndex>>>,

    tcx: TyCtxt<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    body: &'a Body<'tcx>,
    location_map: &'a DenseLocationMap,
    borrow_set: &'a BorrowSet<'tcx>,
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

        let mut constraints = Constraints::new(tcx, regioncx, body, location_map);
        for constraint in regioncx.outlives_constraints() {
            constraints.add_constraint(&constraint);
        }

        Self {
            loans_out_of_scope: IndexVec::from_fn_n(|_| Default::default(), borrow_set.len()),
            loan_scopes: IndexVec::from_elem_n(None, borrow_set.len()),
            constraints,
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

    /// Check if a loan is in scope at a location.
    pub(crate) fn loan_in_scope_at(&mut self, borrow_idx: BorrowIndex, location: Location) -> bool {
        let point = self.location_map.point_from_location(location);
        if let Some(in_scope_points) = &self.loan_scopes[borrow_idx] {
            in_scope_points
        } else {
            let in_scope_points = self.compute_loan_out_of_scope(borrow_idx);
            self.loan_scopes[borrow_idx].insert(in_scope_points)
        }
        .contains(point)
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

                // TODO: Look if this is necessary
                if associated_regions.is_empty() {
                    self.remove_kill(loan_idx, location);
                }

                // Incorporate the added regions into `associated_regions`.
                associated_regions.union(&added_regions);
                my_println!("    Regions: {:?}", associated_regions);

                Some(time_travelling_regions)
            } else {
                my_println!("Nothing new here.");
                if reachable_by_loan {
                    // FIXME: This is just a hack.
                    let mut associated_regions = associated_regions.clone();
                    self.remove_dead_regions(location, &mut associated_regions);
                    if associated_regions.is_empty() {
                        my_println!("  Loan killed.");
                        self.add_kill(loan_idx, location);
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
                if associated_regions.is_empty() {
                    my_println!("  Loan killed.");
                    self.add_kill(loan_idx, location);
                } else if in_scope {
                    in_scope_points.insert(point);
                    my_println!("    In scope at {location:?}");
                }
            }

            // Check if the loan is killed.
            let is_killed = self.kills.get(&location).is_some_and(|x| x.contains(&loan_idx));

            // Update in_scope.
            let successor_in_scope = self.successor_in_scope(loan_idx, location, in_scope);

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
                // FIXME: Only necessary if we record the kills of loans.
                if successor_node.associated_regions.is_empty() {
                    successor_has_changed = true;
                    // That node is killed.
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
                    // FIXME: Only necessary if we record the kills of loans.
                    if successor_node.associated_regions.is_empty() {
                        successor_has_changed = true;
                        // That node is killed.
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

    /// Given the `in_scope` value for a location, return the `in_scope` value for the successor
    /// location(s).
    fn successor_in_scope(
        &self,
        borrow_idx: BorrowIndex,
        location: Location,
        current_in_scope: bool,
    ) -> bool {
        assert_eq!(
            location == self.borrow_set[borrow_idx].reserve_location,
            self.body[location.block]
                .statements
                .get(location.statement_index)
                .is_some_and(|stmt| self.in_scope_at_stmt(borrow_idx, stmt, location))
        );
        if let Some(stmt) = self.body[location.block].statements.get(location.statement_index) {
            let current_in_scope =
                current_in_scope || self.in_scope_at_stmt(borrow_idx, stmt, location);
            current_in_scope && !self.out_of_scope_at_stmt(borrow_idx, stmt)
        } else {
            current_in_scope
                && !self
                    .out_of_scope_at_terminator(borrow_idx, &self.body[location.block].terminator())
        }
    }

    /// Check if a borrow is in scope after this statement, regardless if it was in scope on entry.
    #[inline]
    fn in_scope_at_stmt(
        &self,
        borrow_idx: BorrowIndex,
        stmt: &Statement<'tcx>,
        location: Location,
    ) -> bool {
        if let mir::StatementKind::Assign(box (_lhs, mir::Rvalue::Ref(_, _, place))) = &stmt.kind {
            !place.ignore_borrow(self.tcx, self.body, &self.borrow_set.locals_state_at_exit)
                && borrow_idx == self.borrow_set.get_index_of(&location).unwrap()
        } else {
            false
        }
    }

    /// Given that the borrow was in scope on entry to this statement, check if it goes out of scope
    /// till the next location.
    #[inline]
    fn out_of_scope_at_stmt(&self, borrow_idx: BorrowIndex, stmt: &Statement<'tcx>) -> bool {
        match &stmt.kind {
            mir::StatementKind::Assign(box (lhs, _rhs)) => {
                self.borrow_out_of_scope_on_place(borrow_idx, *lhs)
            }
            mir::StatementKind::StorageDead(local) => {
                self.borrow_out_of_scope_on_place(borrow_idx, Place::from(*local))
            }
            _ => false,
        }
    }

    /// Given that the borrow was in scope on entry to this terminator, check if it goes out of scope
    /// till the succeeding blocks.
    #[inline]
    fn out_of_scope_at_terminator(
        &self,
        borrow_idx: BorrowIndex,
        terminator: &Terminator<'tcx>,
    ) -> bool {
        if let mir::TerminatorKind::InlineAsm { operands, .. } = &terminator.kind {
            operands.iter().any(|op| {
                if let mir::InlineAsmOperand::Out { place: Some(place), .. }
                | mir::InlineAsmOperand::InOut { out_place: Some(place), .. } = op
                {
                    self.borrow_out_of_scope_on_place(borrow_idx, *place)
                } else {
                    false
                }
            })
        } else {
            false
        }
    }

    fn borrow_out_of_scope_on_place(&self, borrow_idx: BorrowIndex, place: Place<'tcx>) -> bool {
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
