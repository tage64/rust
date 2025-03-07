use std::assert_matches::assert_matches;
use std::mem;

use rustc_data_structures::fx::FxHashSet;
use rustc_index::IndexVec;
use rustc_index::bit_set::thin_bit_set::{SparseBitMatrix, ThinBitSet};
use rustc_middle::mir::{
    BasicBlock, Body, Location, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::{TyCtxt, TypeVisitable};
use rustc_mir_dataflow::points::{DenseLocationMap, PointIndex};

use super::{new_empty_region_set, new_region_matrix};
use crate::constraints::OutlivesConstraint;
use crate::type_check::Locations;
use crate::{RegionInferenceContext, RegionVid};

/// Outlives constraints organized by the point in the CFG where they take effect.
///
/// This struct contains all outlives constraints. It internally differentiates between global
/// constraints which are in effect everywhere and local constraints which are only in effect at a
/// certain point. It can retrieve all constraints at a certain point in constant time.
pub(crate) struct Constraints<'a, 'tcx> {
    /// A mapping from points to local outlives constraints, (only active at a single point).
    ///
    /// At point `p` we will store all local outlives constraints which takes effect at `p`. That
    /// means that there sup-region (`'a` in `'a: 'b`) will is checked in `p`. As a consequence,
    /// time travelling constraints travelling backwards in time will be stored at the successor location(s) of the location from `constraint.locations`.
    local_constraints: IndexVec<PointIndex, Vec<LocalConstraint>>,

    /// A list of all outlives constraints that are active at every point in the CFG.
    global_constraints: Vec<GlobalConstraint>,

    tcx: TyCtxt<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    body: &'a Body<'tcx>,
    location_map: &'a DenseLocationMap,
}

/// A global outlives constraint which is active at every point in the CFG.
#[derive(Clone, Copy)]
struct GlobalConstraint {
    /// If we have the constraint `'a: 'b`, then `'a` is the sup and `'b` the sub.
    sup: RegionVid,
    /// If we have the constraint `'a: 'b`, then `'a` is the sup and `'b` the sub.
    sub: RegionVid,
}

/// A local outlives constraint which is only active at a single point in the CFG.
#[derive(Clone, Copy)]
struct LocalConstraint {
    /// If we have the constraint `'a: 'b`, then `'a` is the sup and `'b` the sub.
    sup: RegionVid,
    /// If we have the constraint `'a: 'b`, then `'a` is the sup and `'b` the sub.
    sub: RegionVid,

    /// If and how the constraint travels in time.
    time_travel: Option<(TimeTravelDirection, TimeTravelKind)>,
}

/// A direction for time travelling constraints.
///
/// Most local constraints act on a single location in the CFG, but some constraints flows backwards
/// or forwards to the previous/next location. For instance, if we have the constraint `'a: 'b` at
/// location `l`, and the constraint flows forwards in time, then `'b` is active at the successor of
/// `l` if `'a` is active at `l`. Similarly, if the constraint flows backwards in time, `'b` is
/// active at the predecessor of `l` if `'a` is active at `'a`.
#[derive(Debug, Copy, Clone)]
enum TimeTravelDirection {
    /// The constraint flows backwards in time.
    ///
    /// `'a: 'b` at location `l` means that `'b` is active at the predecessor of `l` if `'a` is
    /// active at `l`.
    Backwards,

    /// The constraint flows forwards in time.
    ///
    /// `'a: 'b` at location `l` means that `'b` is active at the successor of `l` if `'a` is
    /// active at `l`.
    Forwards,
}

/// If a time travelling constraint travels within the same block or across block boundaries.
///
/// The constraint's "location"/point is the point in the CFG where the sup-region is checked. So if
/// we have the constraint `'a: 'b`, the constraint's "location" is the location where `'a` is
/// checked. The "target location" is the location where `'b` becomes active. We will call the
/// "location" "source location" for clarity. Remember that the source and target locations are
/// either the same point, or if the constraint is time travelling, they are adjacent points.
#[derive(Debug, Copy, Clone)]
enum TimeTravelKind {
    /// The constraint travels within the same block.
    ///
    /// Let's assume we have the constraint `'a: 'b`. If the constraint travels backwards in time,
    /// then the location where `'a` is checked cannot be the first location in a block, because
    /// then `'b` would be active in a preceeding block. Similarly, if it travels forwards in time,
    /// `'a` cannot be checked at the terminator.
    IntraBlock,

    /// The constraint travels in time to a preceeding or succeeding block.
    ///
    /// The source and target locations are in different blocks. Since they must be adjacent, it
    /// follows that if the constraint is travelling forwards in time, then the source location is a
    /// terminator and the target location is the first location in a block. Similarly, if the
    /// constraint is travelling backwards in time, the source location is first location of a block
    /// and the target location is a terminator.
    InterBlock {
        /// The block of the target location.
        ///
        /// The statement index of the target location is `0` if the constraint is travelling
        /// forwards in time or the index of the terminator if the constraint is travelling
        /// backwards.
        target_block: BasicBlock,
    },
}

#[derive(Default)]
pub(crate) struct TimeTravellingRegions {
    pub to_prev_stmt: Option<ThinBitSet<RegionVid>>,
    pub to_preceeding_blocks: Option<SparseBitMatrix<BasicBlock, RegionVid>>,
    pub to_next_loc: Option<ThinBitSet<RegionVid>>,
    pub to_succeeding_blocks: Option<SparseBitMatrix<BasicBlock, RegionVid>>,
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

    fn add_to_preceeding_block(
        &mut self,
        regioncx: &RegionInferenceContext<'_>,
        region: RegionVid,
        preceeding_block: BasicBlock,
    ) {
        self.to_preceeding_blocks
            .get_or_insert_with(|| new_region_matrix(regioncx))
            .insert(preceeding_block, region);
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

impl<'a, 'tcx> Constraints<'a, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        regioncx: &'a RegionInferenceContext<'tcx>,
        body: &'a Body<'tcx>,
        location_map: &'a DenseLocationMap,
    ) -> Self {
        Self {
            local_constraints: IndexVec::from_elem_n(vec![], location_map.num_points()),
            global_constraints: vec![],
            tcx,
            regioncx,
            body,
            location_map,
        }
    }

    pub(crate) fn add_constraint(&mut self, constraint: &OutlivesConstraint<'tcx>) {
        match constraint.locations {
            Locations::Single(location) => {
                let (source_location, time_travel) = if let Some(stmt) =
                    self.body[location.block].statements.get(location.statement_index)
                {
                    match self.time_travel_at_statement(&constraint, stmt) {
                        Some(t @ TimeTravelDirection::Forwards) => {
                            (location, Some((t, TimeTravelKind::IntraBlock)))
                        }
                        Some(t @ TimeTravelDirection::Backwards) => (
                            location.successor_within_block(),
                            Some((t, TimeTravelKind::IntraBlock)),
                        ),
                        None => (location, None),
                    }
                } else {
                    debug_assert_eq!(
                        location.statement_index,
                        self.body[location.block].statements.len()
                    );
                    let terminator = self.body[location.block].terminator();
                    match self.time_travel_at_terminator(&constraint, terminator) {
                        Some((t @ TimeTravelDirection::Forwards, target_block)) => {
                            (location, Some((t, TimeTravelKind::InterBlock { target_block })))
                        }
                        Some((t @ TimeTravelDirection::Backwards, source_block)) => (
                            Location { block: source_block, statement_index: 0 },
                            Some((t, TimeTravelKind::InterBlock { target_block: location.block })),
                        ),
                        None => (location, None),
                    }
                };

                let point = self.location_map.point_from_location(source_location);
                self.local_constraints[point].push(LocalConstraint {
                    sup: constraint.sup,
                    sub: constraint.sub,
                    time_travel,
                });
            }
            Locations::All(_) => {
                self.global_constraints
                    .push(GlobalConstraint { sup: constraint.sup, sub: constraint.sub });
            }
        }
    }

    /// Checks if and in which direction a constraint at a statement travels in time.
    fn time_travel_at_statement(
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

    /// Check if/how an outlives constraint travels in time at a terminator.
    ///
    /// Returns an `Option` of the pair `(direction, block)`. Where `direction` is a
    /// `TimeTravelDirection` and `block` is the target or source block of a forwards or backwards
    /// travelling constraint respectively.
    fn time_travel_at_terminator(
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

    /// Given a set of regions at a certain point in the CFG, add all regions induced by outlives
    /// constraints at that point  to the set. Additionally, all regions arising from time
    /// travelling constraints will be collected and returned.
    ///
    /// If we have the set `{'a, 'b}`, and we have the following constraints:
    /// - `'a: 'c`
    /// - `'b: 'd`
    /// - `'d: 'e`
    /// Then `'c`, `'d` and `'e` will be added to the set.
    ///
    /// Also, any time travelling constraints implied by any of these five regions would be
    /// collected and returned in the `TimeTravellingRegions` struct.
    pub(crate) fn add_dependent_regions_at_point(
        &self,
        point: PointIndex,
        regions: &mut ThinBitSet<RegionVid>,
    ) -> TimeTravellingRegions {
        // This function will loop until there are no more regions to add. It will keep a set of
        // regions that has not been considered yet (the `to_check` variable). At each iteration of
        // the main loop, It'll walk through all constraints at this point and all global
        // constraints. Any regions implied from the `to_check` set  will be put in the
        // `to_check_next_round` set. When all constraints has been considered, the `to_check` set
        // will be cleared. It will be swaped with the `to_check_next_round` set, and then the main
        // loop runs again. It'll stop when there are no more regions to check.
        //
        // The time travelling constraints will be treated differently. Regions implied by time
        // travelling constraints will be collected in an instance of the `TimeTravellingRegions`
        // struct.

        let mut to_check = regions.clone();
        let mut to_check_next_round = new_empty_region_set(self.regioncx);
        let mut time_travelling_regions = TimeTravellingRegions::default();

        // Loop till the fixpoint: when there are no more regions to add.
        while !to_check.is_empty() {
            // Loop through all global constraints.
            for constraint in &self.global_constraints {
                if !to_check.contains(constraint.sup) {
                    continue;
                }
                if regions.insert(constraint.sub) {
                    to_check_next_round.insert(constraint.sub);
                }
            }

            // Loop through all local constraints.
            for constraint in &self.local_constraints[point] {
                if !to_check.contains(constraint.sup) {
                    continue;
                }

                // Check if the constraint is travelling in time.
                if let Some((travel_direction, travel_kind)) = constraint.time_travel {
                    match (travel_direction, travel_kind) {
                        (direction, TimeTravelKind::IntraBlock) => time_travelling_regions
                            .add_within_block(self.regioncx, constraint.sub, direction),
                        (
                            TimeTravelDirection::Forwards,
                            TimeTravelKind::InterBlock { target_block },
                        ) => time_travelling_regions.add_to_succeeding_block(
                            self.regioncx,
                            constraint.sub,
                            target_block,
                        ),
                        (
                            TimeTravelDirection::Backwards,
                            TimeTravelKind::InterBlock { target_block },
                        ) => time_travelling_regions.add_to_preceeding_block(
                            self.regioncx,
                            constraint.sub,
                            target_block,
                        ),
                    }

                    // If the region is time travelling we should not add it to
                    // `regions`.
                    continue;
                }

                if regions.insert(constraint.sub) {
                    to_check_next_round.insert(constraint.sub);
                }
            }

            mem::swap(&mut to_check, &mut to_check_next_round);
            to_check_next_round.clear();
        }

        time_travelling_regions
    }

    /// Given a set of regions, add all regions induced by outlives constraints at any point in the
    /// CFG to the set.
    ///
    /// If we have the set `{'a, 'b}`, and we have the following constraints:
    /// - `'a: 'c`
    /// - `'b: 'd`
    /// - `'d: 'e`
    /// Then `'c`, `'d` and `'e` will be added to the set.
    pub(crate) fn add_dependent_regions(&self, regions: &mut ThinBitSet<RegionVid>) {
        // This function will loop until there are no more regions to add. It will keep a set of
        // regions that has not been considered yet (the `to_check` variable). At each iteration of
        // the main loop, It'll walk through all constraints, both global and local. Any regions
        // implied from the `to_check` set  will be put in the `to_check_next_round` set. When all
        // constraints has been considered, the `to_check` set will be cleared. It will be swaped
        // with the `to_check_next_round` set, and then the main loop runs again. It'll stop when
        // there are no more regions to check.
        //
        // The time travelling constraints will not be treated differently in this function.

        let mut to_check = regions.clone();
        let mut to_check_next_round = new_empty_region_set(self.regioncx);

        // Loop till the fixpoint: when there are no more regions to add.
        while !to_check.is_empty() {
            // Loop through all global constraints.
            for constraint in &self.global_constraints {
                if !to_check.contains(constraint.sup) {
                    continue;
                }
                if regions.insert(constraint.sub) {
                    to_check_next_round.insert(constraint.sub);
                }
            }

            // Loop through all local constraints.
            for constraint in self.local_constraints.iter().flatten() {
                if !to_check.contains(constraint.sup) {
                    continue;
                }
                if regions.insert(constraint.sub) {
                    to_check_next_round.insert(constraint.sub);
                }
            }

            mem::swap(&mut to_check, &mut to_check_next_round);
            to_check_next_round.clear();
        }
    }
}
