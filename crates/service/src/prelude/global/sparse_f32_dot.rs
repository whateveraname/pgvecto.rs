use std::borrow::Cow;

use crate::prelude::*;

#[derive(Debug, Clone, Copy)]
pub enum SparseF32Dot {}

impl G for SparseF32Dot {
    type Scalar = F32;
    type Storage = SparseMmap;
    type L2 = F32L2;
    type VectorOwned = SparseF32;
    type VectorRef<'a> = SparseF32Ref<'a>;

    const DISTANCE: Distance = Distance::Dot;
    const KIND: Kind = Kind::SparseF32;

    fn owned_to_ref(vector: &SparseF32) -> SparseF32Ref<'_> {
        SparseF32Ref::from(vector)
    }

    fn ref_to_owned(vector: SparseF32Ref<'_>) -> SparseF32 {
        SparseF32::from(vector)
    }

    fn to_dense(vector: Self::VectorRef<'_>) -> Cow<'_, [F32]> {
        Cow::Owned(vector.to_dense())
    }

    fn distance(lhs: Self::VectorRef<'_>, rhs: Self::VectorRef<'_>) -> F32 {
        super::sparse_f32::dot(lhs, rhs) * (-1.0)
    }

    fn elkan_k_means_normalize(vector: &mut [Self::Scalar]) {
        super::f32::l2_normalize(vector)
    }

    fn elkan_k_means_normalize2(vector: &mut SparseF32) {
        super::sparse_f32::l2_normalize(vector)
    }

    fn elkan_k_means_distance(lhs: &[Self::Scalar], rhs: &[Self::Scalar]) -> F32 {
        super::f32::dot(lhs, rhs).acos()
    }

    fn elkan_k_means_distance2(lhs: Self::VectorRef<'_>, rhs: &[Self::Scalar]) -> F32 {
        super::sparse_f32::dot_2(lhs, rhs).acos()
    }

    fn scalar_quantization_distance(
        _dims: u16,
        _max: &[Self::Scalar],
        _min: &[Self::Scalar],
        _lhs: Self::VectorRef<'_>,
        _rhs: &[u8],
    ) -> F32 {
        unimplemented!()
    }

    fn scalar_quantization_distance2(
        _dims: u16,
        _max: &[Self::Scalar],
        _min: &[Self::Scalar],
        _lhs: &[u8],
        _rhs: &[u8],
    ) -> F32 {
        unimplemented!()
    }

    fn product_quantization_distance(
        _dims: u16,
        _ratio: u16,
        _centroids: &[Self::Scalar],
        _lhs: Self::VectorRef<'_>,
        _rhs: &[u8],
    ) -> F32 {
        unimplemented!()
    }

    fn product_quantization_distance2(
        _dims: u16,
        _ratio: u16,
        _centroids: &[Self::Scalar],
        _lhs: &[u8],
        _rhs: &[u8],
    ) -> F32 {
        unimplemented!()
    }

    fn product_quantization_distance_with_delta(
        _dims: u16,
        _ratio: u16,
        _centroids: &[Self::Scalar],
        _lhs: Self::VectorRef<'_>,
        _rhs: &[u8],
        _delta: &[Self::Scalar],
    ) -> F32 {
        unimplemented!()
    }
}
