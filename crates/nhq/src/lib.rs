#![feature(trait_alias)]
#![allow(clippy::len_without_is_empty)]

pub mod hash_order;

use base::index::*;
use base::operator::*;
use base::scalar::F32;
use base::search::*;
use base::vector::VectorBorrowed;
use common::dir_ops::sync_dir;
use common::mmap_array::MmapArray;
use common::visited_pool::VisitedPool;
use hash_order::generate_min_hash_mapping;
use heapify::{make_heap, pop_heap, push_heap};
use parking_lot::RwLock;
use quantization::operator::OperatorQuantization;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::thread_rng;
use rand::Rng;
use rand::SeedableRng;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::create_dir;
use std::path::Path;
use std::sync::Arc;
use storage::operator::OperatorStorage;
use storage::StorageCollection;

pub trait OperatorNhq = Operator + OperatorQuantization + OperatorStorage;

const CONTROL_NUM: usize = 1000;

pub struct Nhq<O: OperatorNhq> {
    mmap: NhqMmap<O>,
}

impl<O: OperatorNhq> Nhq<O> {
    pub fn create<S: Source<O>>(path: &Path, options: IndexOptions, source: &S) -> Self {
        create_dir(path).unwrap();
        let ram = make(path, options, source);
        let mmap = save(ram, path);
        sync_dir(path);
        Self { mmap }
    }

    pub fn open(path: &Path, options: IndexOptions) -> Self {
        let mmap = open(path, options);
        Self { mmap }
    }

    pub fn basic(
        &self,
        vector: Borrowed<'_, O>,
        opts: &SearchOptions,
        filter: impl Filter,
    ) -> BinaryHeap<Reverse<Element>> {
        basic(&self.mmap, vector, opts.hnsw_ef_search, filter)
    }

    pub fn vbase<'a>(
        &'a self,
        vector: Borrowed<'a, O>,
        opts: &'a SearchOptions,
        filter: impl Filter + 'a,
    ) -> (Vec<Element>, Box<(dyn Iterator<Item = Element> + 'a)>) {
        vbase(&self.mmap, vector, opts.hnsw_ef_search, filter)
    }

    pub fn len(&self) -> u32 {
        self.mmap.storage.len()
    }

    pub fn vector(&self, i: u32) -> Borrowed<'_, O> {
        self.mmap.storage.vector(i)
    }

    pub fn payload(&self, i: u32) -> Payload {
        self.mmap.storage.payload(i)
    }
}

unsafe impl<O: OperatorNhq> Send for Nhq<O> {}
unsafe impl<O: OperatorNhq> Sync for Nhq<O> {}

pub struct NhqRam<O: OperatorNhq> {
    storage: Arc<StorageCollection<O>>,
    // ----------------------
    dims: u32,
    // ----------------------
    m: u32,
    // ----------------------
    graph: Vec<Vec<u32>>,
    // ----------------------
    visited: VisitedPool,
}

pub struct NhqMmap<O: OperatorNhq> {
    storage: Arc<StorageCollection<O>>,
    // ----------------------
    dims: u32,
    // ----------------------
    m: u32,
    // ----------------------
    edges: MmapArray<u32>,
    // ----------------------
    reordered_vectors: MmapArray<Scalar<O>>,
    reordered_payload: MmapArray<Payload>,
    // ----------------------
    visited: VisitedPool,
}

unsafe impl<O: OperatorNhq> Send for NhqMmap<O> {}
unsafe impl<O: OperatorNhq> Sync for NhqMmap<O> {}

pub fn make<O: OperatorNhq, S: Source<O>>(
    path: &Path,
    options: IndexOptions,
    source: &S,
) -> NhqRam<O> {
    let NhqIndexingOptions {
        m,
        k,
        l,
        s,
        r,
        quantization: _,
    } = options.indexing.clone().unwrap_nhq();
    let storage = Arc::new(StorageCollection::create(&path.join("raw"), source));
    rayon::check();
    let visited = VisitedPool::new(storage.len());
    let n = storage.len();
    let mut graph = initialize_graph(storage.clone(), n, l, s);
    let control_points: Vec<u32> = sample(&mut thread_rng(), n as usize, CONTROL_NUM)
        .iter()
        .map(|i| i as u32)
        .collect();
    let acc_eval_set = generate_control_set(storage.clone(), &control_points, n);
    loop {
        update(&mut graph, s, r);
        join(storage.clone(), &mut graph, n);
        if eval_recall(&graph, &control_points, &acc_eval_set, k) > 0.8 {
            break;
        }
    }
    let graph = select_edge(storage.clone(), &mut graph, n, k, m);
    NhqRam {
        storage,
        dims: options.vector.dims,
        m,
        graph,
        visited,
    }
}

pub fn save<O: OperatorNhq>(ram: NhqRam<O>, path: &Path) -> NhqMmap<O> {
    let storage = &ram.storage;
    let n = storage.len();
    let perm = generate_min_hash_mapping(&ram.graph, n, 2, 1, 42);
    // let mut perm = (0..n).collect::<Vec<_>>();
    let mut order = vec![0; n as usize];
    for i in 0..n {
        order[perm[i as usize] as usize] = i;
    }
    let edges = MmapArray::create(
        &path.join("edges"),
        (0..n)
            .map(|i| {
                let mut v = ram.graph[order[i as usize] as usize].clone();
                for j in 0..v.len() {
                    v[j] = perm[v[j] as usize] as u32;
                }
                let len = v.len();
                v.resize_with(ram.m as usize, || 0);
                v.insert(0, len as u32);
                v
            })
            .flat_map(|v| v.into_iter()),
    );
    let vectors_iter = (0..n).flat_map(|i| storage.vector(order[i as usize]).to_vec());
    let payload_iter = (0..n).map(|i| storage.payload(order[i as usize]));
    let vectors = MmapArray::create(&path.join("vectors"), vectors_iter);
    let payload = MmapArray::create(&path.join("payload"), payload_iter);
    NhqMmap {
        storage: ram.storage,
        dims: ram.dims,
        m: ram.m,
        edges,
        reordered_vectors: vectors,
        reordered_payload: payload,
        visited: ram.visited,
    }
}

pub fn open<O: OperatorNhq>(path: &Path, options: IndexOptions) -> NhqMmap<O> {
    let idx_opts = options.indexing.clone().unwrap_nhq();
    let storage = Arc::new(StorageCollection::open(&path.join("raw"), options.clone()));
    let edges = MmapArray::open(&path.join("edges"));
    let n = storage.len();
    let vectors = MmapArray::open(&path.join("vectors"));
    let payload = MmapArray::open(&path.join("payload"));
    NhqMmap {
        storage,
        dims: options.vector.dims,
        m: idx_opts.m,
        edges,
        reordered_vectors: vectors,
        reordered_payload: payload,
        visited: VisitedPool::new(n),
    }
}

pub fn basic<O: OperatorNhq>(
    mmap: &NhqMmap<O>,
    vector: Borrowed<'_, O>,
    ef_search: usize,
    mut filter: impl Filter,
) -> BinaryHeap<Reverse<Element>> {
    let Some(s) = entry(mmap, filter.clone()) else {
        return BinaryHeap::new();
    };
    let dims = mmap.dims as usize;
    let mut visited = mmap.visited.fetch2();
    let mut candidates = BinaryHeap::<Reverse<(F32, u32)>>::new();
    let mut results = ElementHeap::new(ef_search);
    visited.mark(s);
    let s_dis = O::distance2(
        vector,
        &mmap.reordered_vectors[s as usize * dims..(s as usize + 1) * dims],
    );
    candidates.push(Reverse((s_dis, s)));
    results.push(Element {
        distance: s_dis,
        payload: mmap.reordered_payload[s as usize],
    });
    while let Some(Reverse((u_dis, u))) = candidates.pop() {
        if !results.check(u_dis) {
            break;
        }
        let edges = find_edges(mmap, u);
        for &v in edges {
            if !visited.check(v) {
                continue;
            }
            visited.mark(v);
            if !filter.check(mmap.reordered_payload[v as usize]) {
                continue;
            }
            let v_dis = O::distance2(
                vector,
                &mmap.reordered_vectors[v as usize * dims..(v as usize + 1) * dims],
            );
            if !results.check(v_dis) {
                continue;
            }
            candidates.push(Reverse((v_dis, v)));
            results.push(Element {
                distance: v_dis,
                payload: mmap.reordered_payload[v as usize],
            });
        }
    }
    results.into_reversed_heap()
}

pub fn vbase<'a, O: OperatorNhq>(
    mmap: &'a NhqMmap<O>,
    vector: Borrowed<'a, O>,
    range: usize,
    mut filter: impl Filter + 'a,
) -> (Vec<Element>, Box<(dyn Iterator<Item = Element> + 'a)>) {
    let Some(s) = entry(mmap, filter.clone()) else {
        return (Vec::new(), Box::new(std::iter::empty()));
    };
    let dims = mmap.dims as usize;
    let mut visited = mmap.visited.fetch2();
    let mut candidates = BinaryHeap::<Reverse<(F32, u32)>>::new();
    visited.mark(s);
    let s_dis = O::distance2(
        vector,
        &mmap.reordered_vectors[s as usize * dims..(s as usize + 1) * dims],
    );
    candidates.push(Reverse((s_dis, s)));
    let mut results = ElementHeap::new(range);
    results.push(Element {
        distance: s_dis,
        payload: mmap.reordered_payload[s as usize],
    });
    let mut stage1 = 1;
    while let Some(Reverse((u_dis, u))) = candidates.pop() {
        if !results.check(u_dis) {
            if stage1 == 1 {
                stage1 = 0;
            } else {
                break;
            }
        }
        let edges = find_edges(mmap, u);
        for &v in edges
            .iter()
            .take((mmap.m as f32 / (stage1 as f32 + 0.5)) as usize)
        {
            if !visited.check(v) {
                continue;
            }
            visited.mark(v);
            if !filter.check(mmap.reordered_payload[v as usize]) {
                continue;
            }
            let v_dis = O::distance2(
                vector,
                &mmap.reordered_vectors[v as usize * dims..(v as usize + 1) * dims],
            );
            if !results.check(v_dis) {
                continue;
            }
            candidates.push(Reverse((v_dis, v)));
            results.push(Element {
                distance: v_dis,
                payload: mmap.reordered_payload[v as usize],
            });
        }
    }
    let iter = std::iter::from_fn(move || {
        let Reverse((u_dis, u)) = candidates.pop()?;
        {
            let edges = find_edges(mmap, u);
            for &v in edges {
                if !visited.check(v) {
                    continue;
                }
                visited.mark(v);
                if filter.check(mmap.reordered_payload[v as usize]) {
                    let v_dis = O::distance2(
                        vector,
                        &mmap.reordered_vectors[v as usize * dims..(v as usize + 1) * dims],
                    );
                    candidates.push(Reverse((v_dis, v)));
                }
            }
        }
        Some(Element {
            distance: u_dis,
            payload: mmap.reordered_payload[u as usize],
        })
    });
    (results.binary_heap.into_sorted_vec(), Box::new(iter))
}

fn generate_control_set<O: OperatorNhq>(
    storage: Arc<StorageCollection<O>>,
    c: &[u32],
    n: u32,
) -> Vec<Vec<u32>> {
    let mut v = vec![Vec::new(); CONTROL_NUM];
    v.par_iter_mut().enumerate().for_each(|(i, vi)| {
        let mut tmp = BinaryHeap::new();
        for j in 0..n {
            let dist = O::distance(storage.vector(c[i]), storage.vector(j));
            tmp.push(Neighbor::new(j, dist, true));
        }
        *vi = tmp
            .into_sorted_vec()
            .into_iter()
            .take(CONTROL_NUM)
            .map(|neighbor| neighbor.id)
            .collect();
    });
    v
}

fn eval_recall(graph: &[Nhood], control_points: &[u32], acc_eval_set: &[Vec<u32>], k: u32) -> f32 {
    control_points
        .iter()
        .enumerate()
        .map(|(i, &cp)| {
            let mut pool = graph[cp as usize].pool.read().clone();
            pool.sort_unstable();
            let acc = pool
                .iter()
                .take(k as usize)
                .map(|g| {
                    acc_eval_set[i]
                        .iter()
                        .take(k as usize)
                        .find(|&&v| g.id == v)
                        .map_or(0.0, |_| 1.0)
                })
                .sum::<f32>()
                / k as f32;
            acc
        })
        .sum::<f32>()
        / control_points.len() as f32
}

fn initialize_graph<O: OperatorNhq>(
    storage: Arc<StorageCollection<O>>,
    n: u32,
    l: u32,
    s: u32,
) -> Vec<Nhood> {
    let mut graph = Vec::new();
    let mut rng = StdRng::from_entropy();
    graph.resize_with(n as usize, || Nhood::new(l, s, &mut rng, n));
    graph.par_iter_mut().enumerate().for_each(|(i, nhood)| {
        let mut pool = nhood.pool.write();
        let init_neighbors = sample(&mut thread_rng(), n as usize, s as usize + 1).into_vec();
        for id in init_neighbors {
            if id == i {
                continue;
            }
            let dist = O::distance(storage.vector(i as u32), storage.vector(id as u32));
            pool.push(Neighbor::new(id as u32, dist, true));
        }
    });
    graph
}

fn update(graph: &mut Vec<Nhood>, s: u32, r: u32) {
    graph.par_iter_mut().for_each(|nhood| {
        let mut pool = nhood.pool.write();
        pool.sort_unstable();
        let maxl = std::cmp::min(nhood.m + s, pool.len() as u32);
        let mut c = 0;
        let mut l = 0;
        while l < maxl && c < s {
            if pool[l as usize].flag {
                c += 1;
            }
            l += 1;
        }
        nhood.m = l;
    });
    graph.par_iter().enumerate().for_each(|(i, nhood)| {
        let (nn_new, nn_old) = {
            let mut nn_new = Vec::new();
            let mut nn_old = Vec::new();
            let mut pool = nhood.pool.write();
            for nn in pool.iter_mut().take(nhood.m as usize) {
                let nhood_o = &graph[nn.id as usize];
                if nn.flag {
                    nn_new.push(nn.id);
                    if nn.distance > *nhood_o.upper_bound.read() {
                        let mut nhood_o_rnn_new = nhood_o.rnn_new.write();
                        if nhood_o_rnn_new.len() < r as usize {
                            nhood_o_rnn_new.push(i as u32);
                        } else {
                            let pos = thread_rng().gen_range(0..r);
                            nhood_o_rnn_new[pos as usize] = i as u32;
                        }
                    }
                    nn.flag = false;
                } else {
                    nn_old.push(nn.id);
                    if nn.distance > *nhood_o.upper_bound.read() {
                        let mut nhood_o_rnn_old = nhood_o.rnn_old.write();
                        if nhood_o_rnn_old.len() < r as usize {
                            nhood_o_rnn_old.push(i as u32);
                        } else {
                            let pos = thread_rng().gen_range(0..r);
                            nhood_o_rnn_old[pos as usize] = i as u32;
                        }
                    }
                }
            }
            make_heap(&mut pool);
            (nn_new, nn_old)
        };
        *nhood.nn_new.write() = nn_new;
        *nhood.nn_old.write() = nn_old;
    });
    graph.par_iter_mut().for_each(|nhood| {
        let mut nn_new = nhood.nn_new.write();
        let mut nn_old = nhood.nn_old.write();
        let mut rnn_new = nhood.rnn_new.write();
        let mut rnn_old = nhood.rnn_old.write();
        if r != 0 && rnn_new.len() > r as usize {
            rnn_new.shuffle(&mut thread_rng());
            rnn_new.truncate(r as usize);
            rnn_new.shrink_to_fit();
        }
        nn_new.extend(rnn_new.drain(..));
        if r != 0 && rnn_old.len() > r as usize {
            rnn_old.shuffle(&mut thread_rng());
            rnn_old.truncate(r as usize);
            rnn_old.shrink_to_fit();
        }
        nn_old.extend(rnn_old.drain(..));
        if nn_old.len() > r as usize * 2 {
            nn_old.truncate(r as usize * 2);
            nn_old.shrink_to_fit();
        }
    });
}

fn join<O: OperatorNhq>(storage: Arc<StorageCollection<O>>, graph: &mut [Nhood], n: u32) {
    (0..n).into_par_iter().for_each(|id| {
        graph[id as usize].join(|i, j| {
            if i != j {
                let dist = O::distance(storage.vector(i), storage.vector(j));
                graph[i as usize].insert(j, dist);
                graph[j as usize].insert(i, dist);
            }
        });
    });
}

fn select_edge<O: OperatorNhq>(
    storage: Arc<StorageCollection<O>>,
    graph: &mut [Nhood],
    n: u32,
    k: u32,
    m: u32,
) -> Vec<Vec<u32>> {
    let mut final_graph = vec![Vec::new(); n as usize];
    final_graph
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, node)| {
            let mut pool = graph[i].pool.write();
            pool.sort_unstable();
            for Neighbor {
                id: u,
                distance: u_dis,
                flag: _,
            } in pool.iter().take(k as usize)
            {
                if node.len() == m as usize {
                    break;
                }
                let check = node
                    .iter()
                    .map(|&v| O::distance(storage.vector(*u), storage.vector(v)))
                    .all(|dist| dist > *u_dis);
                if check {
                    node.push(*u);
                }
            }
        });
    final_graph
}

pub fn entry<O: OperatorNhq>(mmap: &NhqMmap<O>, mut filter: impl Filter) -> Option<u32> {
    let m = mmap.m;
    let n = mmap.storage.len();
    let mut shift = 1u64;
    let mut count = 0u64;
    while shift * m as u64 <= n as u64 {
        shift *= m as u64;
    }
    while shift != 0 {
        let mut i = 1u64;
        while i * shift <= n as u64 {
            let e = (i * shift - 1) as u32;
            if i % m as u64 != 0 {
                if filter.check(mmap.reordered_payload[e as usize]) {
                    return Some(e);
                }
                count += 1;
                if count >= 10000 {
                    return None;
                }
            }
            i += 1;
        }
        shift /= m as u64;
    }
    None
}

fn find_edges<O: OperatorNhq>(mmap: &NhqMmap<O>, u: u32) -> &[u32] {
    let s = u as usize * (mmap.m as usize + 1) + 1;
    let e = s + mmap.edges[s - 1] as usize;
    &mmap.edges[s..e]
}

#[derive(Default, Clone, Eq)]
struct Neighbor {
    id: u32,
    distance: F32,
    flag: bool,
}

impl Neighbor {
    fn new(id: u32, distance: F32, flag: bool) -> Self {
        Neighbor { id, distance, flag }
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

struct Nhood {
    m: u32,
    upper_bound: RwLock<F32>,
    pool: RwLock<Vec<Neighbor>>,
    nn_old: RwLock<Vec<u32>>,
    nn_new: RwLock<Vec<u32>>,
    rnn_old: RwLock<Vec<u32>>,
    rnn_new: RwLock<Vec<u32>>,
}

impl Nhood {
    fn new(l: u32, s: u32, rng: &mut StdRng, n: u32) -> Self {
        let nn_new = RwLock::new(
            sample(rng, n as usize, s as usize * 2)
                .iter()
                .map(|i| i as u32)
                .collect(),
        );
        Nhood {
            m: s,
            upper_bound: RwLock::new(F32(0.0)),
            pool: RwLock::new(Vec::with_capacity(l as usize)),
            nn_old: RwLock::new(Vec::new()),
            nn_new,
            rnn_old: RwLock::new(Vec::new()),
            rnn_new: RwLock::new(Vec::new()),
        }
    }

    fn insert(&self, id: u32, dist: F32) {
        {
            let pool = self.pool.read();
            if dist > pool[0].distance {
                return;
            }
            if pool.iter().any(|neighbor| neighbor.id == id) {
                return;
            }
        }
        {
            let mut pool = self.pool.write();
            if pool.len() < pool.capacity() {
                pool.push(Neighbor::new(id, dist, true));
                push_heap(&mut pool);
            } else {
                pop_heap(&mut pool);
                let len = pool.len();
                pool[len - 1] = Neighbor::new(id, dist, true);
                push_heap(&mut pool);
            }
            let mut upper_bound = self.upper_bound.write();
            *upper_bound = pool[0].distance;
        }
    }

    fn join<C>(&self, mut callback: C)
    where
        C: FnMut(u32, u32),
    {
        let nn_new = self.nn_new.read();
        let nn_old = self.nn_old.read();
        for &i in &*nn_new {
            for &j in &*nn_new {
                if i < j {
                    callback(i, j);
                }
            }
            for &j in &*nn_old {
                callback(i, j);
            }
        }
    }
}

pub struct ElementHeap {
    binary_heap: BinaryHeap<Element>,
    k: usize,
}

impl ElementHeap {
    pub fn new(k: usize) -> Self {
        assert!(k != 0);
        Self {
            binary_heap: BinaryHeap::new(),
            k,
        }
    }
    pub fn check(&self, distance: F32) -> bool {
        self.binary_heap.len() < self.k || distance < self.binary_heap.peek().unwrap().distance
    }
    pub fn push(&mut self, element: Element) -> Option<Element> {
        self.binary_heap.push(element);
        if self.binary_heap.len() == self.k + 1 {
            self.binary_heap.pop()
        } else {
            None
        }
    }
    pub fn into_reversed_heap(self) -> BinaryHeap<Reverse<Element>> {
        self.binary_heap.into_iter().map(Reverse).collect()
    }
}
