use std::path::Path;
use base::index::{NhqIndexingOptions, IndexOptions, OptimizingOptions, SegmentsOptions, VectorOptions};

fn main() {
    let path = Path::new("/home/yanqi/stand-alone-test/data/nhq_hashorder");
    let options = IndexOptions {
        vector: VectorOptions {
            dims: 128,
            v: base::vector::VectorKind::Vecf32,
            d: base::distance::DistanceKind::L2,
        },
        segment: SegmentsOptions::default(),
        optimizing: OptimizingOptions::default(),
        indexing: base::index::IndexingOptions::Nhq(NhqIndexingOptions::default()),
    };
    nhq::mock_create(path, options);
}