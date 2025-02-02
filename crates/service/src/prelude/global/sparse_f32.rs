use crate::prelude::*;

#[inline(always)]
#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn cosine<'a>(lhs: SparseF32Ref<'a>, rhs: SparseF32Ref<'a>) -> F32 {
    let mut lhs_pos = 0;
    let mut rhs_pos = 0;
    let size1 = lhs.length() as usize;
    let size2 = rhs.length() as usize;
    let mut xy = F32::zero();
    let mut x2 = F32::zero();
    let mut y2 = F32::zero();
    while lhs_pos < size1 && rhs_pos < size2 {
        let lhs_index = lhs.indexes[lhs_pos];
        let rhs_index = rhs.indexes[rhs_pos];
        let lhs_value = lhs.values[lhs_pos];
        let rhs_value = rhs.values[rhs_pos];
        xy += F32((lhs_index == rhs_index) as u32 as f32) * lhs_value * rhs_value;
        x2 += F32((lhs_index <= rhs_index) as u32 as f32) * lhs_value * lhs_value;
        y2 += F32((lhs_index >= rhs_index) as u32 as f32) * rhs_value * rhs_value;
        lhs_pos += (lhs_index <= rhs_index) as usize;
        rhs_pos += (lhs_index >= rhs_index) as usize;
    }
    for i in lhs_pos..size1 {
        x2 += lhs.values[i] * lhs.values[i];
    }
    for i in rhs_pos..size2 {
        y2 += rhs.values[i] * rhs.values[i];
    }
    xy / (x2 * y2).sqrt()
}

#[inline(always)]
#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn dot<'a>(lhs: SparseF32Ref<'a>, rhs: SparseF32Ref<'a>) -> F32 {
    let mut lhs_pos = 0;
    let mut rhs_pos = 0;
    let size1 = lhs.length() as usize;
    let size2 = rhs.length() as usize;
    let mut xy = F32::zero();
    while lhs_pos < size1 && rhs_pos < size2 {
        let lhs_index = lhs.indexes[lhs_pos];
        let rhs_index = rhs.indexes[rhs_pos];
        let lhs_value = lhs.values[lhs_pos];
        let rhs_value = rhs.values[rhs_pos];
        xy += F32((lhs_index == rhs_index) as u32 as f32) * lhs_value * rhs_value;
        lhs_pos += (lhs_index <= rhs_index) as usize;
        rhs_pos += (lhs_index >= rhs_index) as usize;
    }
    xy
}

#[inline(always)]
#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn dot_2<'a>(lhs: SparseF32Ref<'a>, rhs: &[F32]) -> F32 {
    let mut xy = F32::zero();
    for i in 0..lhs.indexes.len() {
        xy += lhs.values[i] * rhs[lhs.indexes[i] as usize];
    }
    xy
}

#[inline(always)]
#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn sl2<'a>(lhs: SparseF32Ref<'a>, rhs: SparseF32Ref<'a>) -> F32 {
    let mut lhs_pos = 0;
    let mut rhs_pos = 0;
    let size1 = lhs.length() as usize;
    let size2 = rhs.length() as usize;
    let mut d2 = F32::zero();
    while lhs_pos < size1 && rhs_pos < size2 {
        let lhs_index = lhs.indexes[lhs_pos];
        let rhs_index = rhs.indexes[rhs_pos];
        let lhs_value = lhs.values[lhs_pos];
        let rhs_value = rhs.values[rhs_pos];
        let d = F32((lhs_index <= rhs_index) as u32 as f32) * lhs_value
            - F32((lhs_index >= rhs_index) as u32 as f32) * rhs_value;
        d2 += d * d;
        lhs_pos += (lhs_index <= rhs_index) as usize;
        rhs_pos += (lhs_index >= rhs_index) as usize;
    }
    for i in lhs_pos..size1 {
        d2 += lhs.values[i] * lhs.values[i];
    }
    for i in rhs_pos..size2 {
        d2 += rhs.values[i] * rhs.values[i];
    }
    d2
}

#[inline(always)]
#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn sl2_2<'a>(lhs: SparseF32Ref<'a>, rhs: &[F32]) -> F32 {
    let mut d2 = F32::zero();
    let mut lhs_pos: u16 = 0;
    let mut rhs_pos: u16 = 0;
    while lhs_pos < lhs.length() {
        let index_eq = lhs.indexes[lhs_pos as usize] == rhs_pos;
        let d = F32(index_eq as u32 as f32) * lhs.values[lhs_pos as usize] - rhs[rhs_pos as usize];
        d2 += d * d;
        lhs_pos += index_eq as u16;
        rhs_pos += 1;
    }
    for i in rhs_pos..rhs.len() as u16 {
        d2 += rhs[i as usize] * rhs[i as usize];
    }
    d2
}

#[inline(always)]
#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn length<'a>(vector: SparseF32Ref<'a>) -> F32 {
    let mut dot = F32::zero();
    for &i in vector.values {
        dot += i * i;
    }
    dot.sqrt()
}

#[inline(always)]
#[multiversion::multiversion(targets(
    "x86_64/x86-64-v4",
    "x86_64/x86-64-v3",
    "x86_64/x86-64-v2",
    "aarch64+neon"
))]
pub fn l2_normalize(vector: &mut SparseF32) {
    let l = length(SparseF32Ref::from(vector as &SparseF32));
    for i in vector.values.iter_mut() {
        *i /= l;
    }
}
