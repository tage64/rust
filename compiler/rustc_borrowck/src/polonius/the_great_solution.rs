#![allow(dead_code)]
mod constraints;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

use constraints::Constraints;
use rustc_data_structures::fx::{FxHashMap, FxIndexMap};
use rustc_index::bit_set::{DenseBitSet, SparseBitMatrix};
use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::{BasicBlock, Body, Location};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::points::DenseLocationMap;

use super::ConstraintDirection;
use super::loan_liveness::collect_kills;
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

    /// All outlives constraints.
    constraints: Constraints<'a, 'tcx>,

    /// A mapping from locations in the CFG to a set of loans that go out of scope. This will be the
    /// final result of the computation.
    loans_out_of_scope: IndexVec<BorrowIndex, FxIndexMap<PoloniusBlock, BTreeSet<usize>>>,

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
    associated_regions: DenseBitSet<RegionVid>,
    added_regions: Option<DenseBitSet<RegionVid>>,
    reachable_by_loan: bool,
    added_to_stack: bool,
}

impl<'a, 'tcx> PoloniusOutOfScopePrecomputer<'a, 'tcx> {
    fn new(
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
            let time_travelling_regions = self.constraints.add_dependent_regions_at_point(
                self.location_map.point_from_location(location),
                &mut added_regions,
            );
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
                    self.remove_dead_regions(location, &mut forward_regions);
                    self.remove_dead_regions(successor_location, &mut forward_regions);
                    if let Some(time_travellers) = &time_travelling_regions.to_next_loc {
                        forward_regions.union(time_travellers);
                    }
                    forward_regions.subtract(&successor_node.associated_regions);
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
                        self.remove_dead_regions(location, &mut forward_regions);
                        self.remove_dead_regions(successor_location, &mut forward_regions);
                        if let Some(time_travellers) = time_travelling_regions
                            .to_succeeding_blocks
                            .as_ref()
                            .and_then(|x| x.row(successor_block))
                        {
                            forward_regions.union(time_travellers);
                        }
                        forward_regions.subtract(&successor_node.associated_regions);
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
                // To comply with previous Polonius, this if condition was:
                // `!is_killed || !location.is_predecessor_of(predecessor_location)`
                // But it doesn't seem to be needed to pass the tests.
                if !is_killed {
                    self.remove_dead_regions(location, &mut backward_regions);
                    self.remove_dead_regions(predecessor_location, &mut backward_regions);
                    if let Some(time_travellers) = &time_travelling_regions.to_prev_stmt {
                        backward_regions.union(time_travellers);
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
                            added_to_stack: false,
                        });
                    if !is_killed {
                        self.remove_dead_regions(location, &mut backward_regions);
                        self.remove_dead_regions(predecessor_location, &mut backward_regions);
                        if let Some(time_travellers) = time_travelling_regions
                            .to_preceeding_blocks
                            .as_ref()
                            .and_then(|x| x.row(predecessor_block))
                        {
                            backward_regions.union(time_travellers);
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
    }

    /// Remove dead regions from the set of associated regions.
    fn remove_dead_regions(&self, location: Location, region_set: &mut DenseBitSet<RegionVid>) {
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
