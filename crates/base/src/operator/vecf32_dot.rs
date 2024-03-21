use crate::distance::*;
use crate::operator::*;
use crate::scalar::*;
use crate::vector::*;

#[derive(Debug, Clone, Copy)]
pub enum Vecf32Dot {}

impl Operator for Vecf32Dot {
    type VectorOwned = Vecf32Owned;

    const DISTANCE_KIND: DistanceKind = DistanceKind::Dot;

    fn distance(lhs: Vecf32Borrowed<'_>, rhs: Vecf32Borrowed<'_>) -> F32 {
        vecf32::dot(lhs.slice(), rhs.slice()) * (-1.0)
    }
    fn distance2(lhs: Borrowed<'_, Self>, rhs: &[Scalar<Self>]) -> F32 {
        vecf32::dot(lhs.slice(), rhs) * (-1.0)
    }
}
