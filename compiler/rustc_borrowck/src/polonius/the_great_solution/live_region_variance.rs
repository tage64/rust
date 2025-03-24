use std::collections::BTreeMap;

use rustc_middle::mir::visit::{TyContext, Visitor};
use rustc_middle::mir::{Body, Location, SourceInfo};
use rustc_middle::ty::relate::{Relate, TypeRelation};
use rustc_middle::ty::visit::TypeVisitable;
use rustc_middle::ty::{GenericArgsRef, Region, RegionVid, Ty, TyCtxt};
use rustc_middle::{span_bug, ty};

use super::super::liveness_constraints::VarianceExtractor;
use super::ConstraintDirection;
use crate::RegionInferenceContext;

/// Some variables are "regular live" at `location` -- i.e., they may be used later. This means that
/// all regions appearing in their type must be live at `location`.
pub(super) fn compute_live_region_variances<'tcx>(
    tcx: TyCtxt<'tcx>,
    regioncx: &RegionInferenceContext<'tcx>,
    body: &Body<'tcx>,
) -> BTreeMap<RegionVid, ConstraintDirection> {
    let mut directions = BTreeMap::new();

    let variance_extractor = VarianceExtractor {
        tcx,
        ambient_variance: ty::Variance::Covariant,
        directions: &mut directions,
        universal_regions: regioncx.universal_regions(),
    };

    let mut visitor = LiveVariablesVisitor { tcx, regioncx, variance_extractor };

    for (bb, data) in body.basic_blocks.iter_enumerated() {
        visitor.visit_basic_block_data(bb, data);
    }

    directions
}

struct LiveVariablesVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    variance_extractor: VarianceExtractor<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for LiveVariablesVisitor<'a, 'tcx> {
    /// We sometimes have `args` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_args(&mut self, args: &GenericArgsRef<'tcx>, _: Location) {
        self.record_regions_live_at(*args);
        self.super_args(args);
    }

    /// We sometimes have `region`s within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_region(&mut self, region: Region<'tcx>, _: Location) {
        self.record_regions_live_at(region);
        self.super_region(region);
    }

    /// We sometimes have `ty`s within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_ty(&mut self, ty: Ty<'tcx>, ty_context: TyContext) {
        match ty_context {
            TyContext::ReturnTy(SourceInfo { span, .. })
            | TyContext::YieldTy(SourceInfo { span, .. })
            | TyContext::ResumeTy(SourceInfo { span, .. })
            | TyContext::UserTy(span)
            | TyContext::LocalDecl { source_info: SourceInfo { span, .. }, .. } => {
                span_bug!(span, "should not be visiting outside of the CFG: {:?}", ty_context);
            }
            TyContext::Location(_) => {
                self.record_regions_live_at(ty);
            }
        }

        self.super_ty(ty);
    }
}

impl<'a, 'tcx> LiveVariablesVisitor<'a, 'tcx> {
    /// Some variable is "regular live" at `location` -- i.e., it may be used later. This means that
    /// all regions appearing in the type of `value` must be live at `location`.
    fn record_regions_live_at<T>(&mut self, value: T)
    where
        T: TypeVisitable<TyCtxt<'tcx>> + Relate<TyCtxt<'tcx>>,
    {
        self.variance_extractor
            .relate(value, value)
            .expect("Can't have a type error relating to itself");
    }
}
