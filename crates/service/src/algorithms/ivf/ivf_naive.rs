use crate::algorithms::clustering::elkan_k_means::ElkanKMeans;
use crate::algorithms::quantization::Quantization;
use crate::algorithms::raw::Raw;
use crate::index::indexing::ivf::IvfIndexingOptions;
use crate::index::segments::growing::GrowingSegment;
use crate::index::segments::sealed::SealedSegment;
use crate::index::IndexOptions;
use crate::index::SearchOptions;
use crate::index::VectorOptions;
use crate::prelude::*;
use crate::utils::cells::SyncUnsafeCell;
use crate::utils::dir_ops::sync_dir;
use crate::utils::element_heap::ElementHeap;
use crate::utils::mmap_array::MmapArray;
use crate::utils::vec2::Vec2;
use rand::seq::index::sample;
use rand::thread_rng;
use rayon::current_num_threads;
use rayon::current_thread_index;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::create_dir;
use std::path::PathBuf;
use std::sync::Arc;

pub struct IvfNaive<S: G> {
    mmap: IvfMmap<S>,
}

impl<S: G> IvfNaive<S> {
    pub fn create(
        path: PathBuf,
        options: IndexOptions,
        sealed: Vec<Arc<SealedSegment<S>>>,
        growing: Vec<Arc<GrowingSegment<S>>>,
    ) -> Self {
        create_dir(&path).unwrap();
        let ram = make(path.clone(), sealed, growing, options);
        let mmap = save(ram, path.clone());
        sync_dir(&path);
        Self { mmap }
    }

    pub fn open(path: PathBuf, options: IndexOptions) -> Self {
        let mmap = load(path.clone(), options);
        Self { mmap }
    }

    pub fn len(&self) -> u32 {
        self.mmap.raw.len()
    }

    pub fn vector(&self, i: u32) -> &[S::Scalar] {
        self.mmap.raw.vector(i)
    }

    pub fn payload(&self, i: u32) -> Payload {
        self.mmap.raw.payload(i)
    }

    pub fn basic(
        &self,
        vector: &[S::Scalar],
        opts: &SearchOptions,
        filter: impl Filter,
    ) -> BinaryHeap<Reverse<Element>> {
        basic(&self.mmap, vector, opts.ivf_nprobe, filter)
    }

    pub fn vbase<'a>(
        &'a self,
        vector: &'a [S::Scalar],
        opts: &'a SearchOptions,
        filter: impl Filter + 'a,
    ) -> (Vec<Element>, Box<(dyn Iterator<Item = Element> + 'a)>) {
        vbase(&self.mmap, vector, opts.ivf_nprobe, filter)
    }
}

unsafe impl<S: G> Send for IvfNaive<S> {}
unsafe impl<S: G> Sync for IvfNaive<S> {}

pub struct IvfRam<S: G> {
    raw: Arc<Raw<S>>,
    quantization: Quantization<S>,
    // ----------------------
    dims: u16,
    // ----------------------
    nlist: u32,
    // ----------------------
    centroids: Vec2<S>,
    ptr: Vec<usize>,
    payloads: Vec<Payload>,
}

unsafe impl<S: G> Send for IvfRam<S> {}
unsafe impl<S: G> Sync for IvfRam<S> {}

pub struct IvfMmap<S: G> {
    raw: Arc<Raw<S>>,
    quantization: Quantization<S>,
    // ----------------------
    dims: u16,
    // ----------------------
    nlist: u32,
    // ----------------------
    centroids: MmapArray<S::Scalar>,
    ptr: MmapArray<usize>,
    payloads: MmapArray<Payload>,
}

unsafe impl<S: G> Send for IvfMmap<S> {}
unsafe impl<S: G> Sync for IvfMmap<S> {}

impl<S: G> IvfMmap<S> {
    fn centroids(&self, i: u32) -> &[S::Scalar] {
        let s = i as usize * self.dims as usize;
        let e = (i + 1) as usize * self.dims as usize;
        &self.centroids[s..e]
    }
}

pub fn make<S: G>(
    path: PathBuf,
    sealed: Vec<Arc<SealedSegment<S>>>,
    growing: Vec<Arc<GrowingSegment<S>>>,
    options: IndexOptions,
) -> IvfRam<S> {
    let VectorOptions { dims, .. } = options.vector;
    let IvfIndexingOptions {
        least_iterations,
        iterations,
        nlist,
        nsample,
        quantization: quantization_opts,
    } = options.indexing.clone().unwrap_ivf();
    let raw = Arc::new(Raw::create(
        path.join("raw"),
        options.clone(),
        sealed,
        growing,
    ));
    let quantization = Quantization::create(
        path.join("quantization"),
        options.clone(),
        quantization_opts,
        &raw,
    );
    let n = raw.len();
    let m = std::cmp::min(nsample, n);
    let f = sample(&mut thread_rng(), n as usize, m as usize).into_vec();
    let samples = SyncUnsafeCell::new(Vec2::<S>::new(dims, m as usize));
    (0..m as usize).into_par_iter().for_each(|i| unsafe {
        (&mut *samples.get())[i].copy_from_slice(raw.vector(f[i] as u32));
        S::elkan_k_means_normalize(&mut (&mut *samples.get())[i]);
    });
    let samples = samples.get_ref().clone();
    let mut k_means = ElkanKMeans::new(nlist as usize, samples);
    for _ in 0..least_iterations {
        k_means.iterate();
    }
    for _ in least_iterations..iterations {
        if k_means.iterate() {
            break;
        }
    }
    let centroids = k_means.finish();
    let idx = SyncUnsafeCell::new(vec![0usize; n as usize]);
    (0..n).into_par_iter().for_each(|i| {
        let mut vector = raw.vector(i).to_vec();
        S::elkan_k_means_normalize(&mut vector);
        let mut result = (F32::infinity(), 0);
        for i in 0..nlist {
            let dis = S::elkan_k_means_distance(&vector, &centroids[i as usize]);
            result = std::cmp::min(result, (dis, i));
        }
        unsafe {
            (&mut *idx.get())[i as usize] = result.1 as usize;
        }
    });
    let mut invlists_payloads = SyncUnsafeCell::new(vec![Vec::new(); nlist as usize]);
    let invlists_codes = SyncUnsafeCell::new(vec![Vec::new(); nlist as usize]);
    (0..current_num_threads()).into_par_iter().for_each(|_| {
        let thread_id = current_thread_index().unwrap();
        let thread_num = current_num_threads();
        for i in 0..n {
            let centroid_id = idx.get_ref()[i as usize];
            let vector = raw.vector(i);
            if centroid_id % thread_num == thread_id {
                unsafe {
                    (&mut *invlists_payloads.get())[centroid_id as usize].push(raw.payload(i));
                    (&mut *invlists_codes.get())[centroid_id as usize].append(&mut vector.to_vec());
                }
            }
        }
    });
    let mut ptr = vec![0usize; nlist as usize + 1];
    let mut payloads = Vec::new();
    for i in 0..nlist {
        ptr[i as usize + 1] = ptr[i as usize] + invlists_payloads.get_ref()[i as usize].len();
        payloads.append(&mut invlists_payloads.get_mut()[i as usize]);
    }
    match &quantization {
        Quantization::Trivial(quantization) => {
            let mut codes = Vec::new();
            for i in 0..nlist {
                codes.append(&mut quantization.codes(i).to_vec());
            }
            MmapArray::create(path.join("vectors"), codes.iter().copied());
        }
        Quantization::Scalar(quantization) => {
            let mut codes = Vec::new();
            for i in 0..nlist {
                codes.append(&mut quantization.codes(i).to_vec());
            }
            MmapArray::create(path.join("vectors"), codes.iter().copied());
        }
        Quantization::Product(quantization) => {
            let mut codes = Vec::new();
            for i in 0..nlist {
                codes.append(&mut quantization.codes(i).to_vec());
            }
            MmapArray::create(path.join("vectors"), codes.iter().copied());
        }
    }
    IvfRam {
        raw,
        quantization,
        centroids,
        ptr,
        payloads,
        nlist,
        dims,
    }
}

pub fn save<S: G>(ram: IvfRam<S>, path: PathBuf) -> IvfMmap<S> {
    let centroids = MmapArray::create(
        path.join("centroids"),
        (0..ram.nlist)
            .flat_map(|i| &ram.centroids[i as usize])
            .copied(),
    );
    let ptr = MmapArray::create(path.join("ptr"), ram.ptr.iter().copied());
    let payloads = MmapArray::create(path.join("payload"), ram.payloads.iter().copied());
    IvfMmap {
        raw: ram.raw,
        quantization: ram.quantization,
        dims: ram.dims,
        nlist: ram.nlist,
        centroids,
        ptr,
        payloads,
    }
}

pub fn load<S: G>(path: PathBuf, options: IndexOptions) -> IvfMmap<S> {
    let raw = Arc::new(Raw::open(path.join("raw"), options.clone()));
    let mut quantization = Quantization::open(
        path.join("quantization"),
        options.clone(),
        options.indexing.clone().unwrap_ivf().quantization,
        &raw,
    );
    let centroids = MmapArray::open(path.join("centroids"));
    let ptr = MmapArray::open(path.join("ptr"));
    let payloads = MmapArray::open(path.join("payload"));
    match &mut quantization {
        Quantization::Trivial(quantization) => {
            let raw = Arc::new(Raw::open(path, options.clone()));
            quantization.set_codes(raw);
        }
        Quantization::Scalar(quantization) => {
            let codes = MmapArray::open(path.join("vectors"));
            quantization.set_codes(codes);
        }
        Quantization::Product(quantization) => {
            let codes = MmapArray::open(path.join("vectors"));
            quantization.set_codes(codes);
        }
    }
    let IvfIndexingOptions { nlist, .. } = options.indexing.unwrap_ivf();
    IvfMmap {
        raw,
        quantization,
        dims: options.vector.dims,
        nlist,
        centroids,
        ptr,
        payloads,
    }
}

pub fn basic<S: G>(
    mmap: &IvfMmap<S>,
    vector: &[S::Scalar],
    nprobe: u32,
    mut filter: impl Filter,
) -> BinaryHeap<Reverse<Element>> {
    let mut target = vector.to_vec();
    S::elkan_k_means_normalize(&mut target);
    let mut lists = ElementHeap::new(nprobe as usize);
    for i in 0..mmap.nlist {
        let centroid = mmap.centroids(i);
        let distance = S::elkan_k_means_distance(&target, centroid);
        if lists.check(distance) {
            lists.push(Element {
                distance,
                payload: i as Payload,
            });
        }
    }
    let lists = lists.into_sorted_vec();
    let mut result = BinaryHeap::new();
    for i in lists.iter().map(|e| e.payload as usize) {
        let start = mmap.ptr[i];
        let end = mmap.ptr[i + 1];
        for j in start..end {
            let payload = mmap.payloads[j];
            if filter.check(payload) {
                let distance = mmap.quantization.distance(vector, j as u32);
                result.push(Reverse(Element { distance, payload }));
            }
        }
    }
    result
}

pub fn vbase<'a, S: G>(
    mmap: &'a IvfMmap<S>,
    vector: &'a [S::Scalar],
    nprobe: u32,
    mut filter: impl Filter + 'a,
) -> (Vec<Element>, Box<(dyn Iterator<Item = Element> + 'a)>) {
    let mut target = vector.to_vec();
    S::elkan_k_means_normalize(&mut target);
    let mut lists = ElementHeap::new(nprobe as usize);
    for i in 0..mmap.nlist {
        let centroid = mmap.centroids(i);
        let distance = S::elkan_k_means_distance(&target, centroid);
        if lists.check(distance) {
            lists.push(Element {
                distance,
                payload: i as Payload,
            });
        }
    }
    let lists = lists.into_sorted_vec();
    let mut result = Vec::new();
    for i in lists.iter().map(|e| e.payload as usize) {
        let start = mmap.ptr[i];
        let end = mmap.ptr[i + 1];
        for j in start..end {
            let payload = mmap.payloads[j];
            if filter.check(payload) {
                let distance = mmap.quantization.distance(vector, j as u32);
                result.push(Element { distance, payload });
            }
        }
    }
    (result, Box::new(std::iter::empty()))
}
