use murmurhash3::murmurhash3_x86_32;
use std::cmp::min;
use std::collections::{BinaryHeap, VecDeque};

pub fn generate_min_hash_mapping(
    graph: &[Vec<u32>],
    n: u32,
    hashes: usize,
    in_bucket: usize,
    seed: u32,
) -> Vec<usize> {
    let num_vertices = n;
    let mut reverse_graph = vec![Vec::new(); num_vertices as usize];
    for v in 0..num_vertices {
        for &u in &graph[v as usize] {
            reverse_graph[u as usize].push(v);
        }
    }
    let mut new_ids = vec![0; num_vertices as usize];

    let mut hash_code1 = vec![0u32; num_vertices as usize];
    let mut hash_code2 = vec![0u32; num_vertices as usize];

    let mask = match hashes {
        1 => u32::MAX,
        2 => u16::MAX as u32,
        _ => u8::MAX as u32,
    };

    for v in 0..num_vertices {
        hash_code1[v as usize] = murmurhash3_x86_32(&v.to_ne_bytes(), seed);
    }

    let mut min_hash;
    let mut min_hash2;
    let mut min_hash3;
    let mut min_hash4;
    if hashes == 1 {
        for v in 0..num_vertices {
            min_hash = u32::MAX;
            for &u in &reverse_graph[v as usize] {
                min_hash = min(hash_code1[u as usize], min_hash);
            }
            hash_code2[v as usize] = min_hash;
        }
    } else if hashes == 2 {
        for v in 0..num_vertices {
            min_hash = u16::MAX as u32;
            min_hash2 = u16::MAX as u32;
            for &u in &reverse_graph[v as usize] {
                min_hash = min(hash_code1[u as usize] >> 16, min_hash);
                min_hash2 = min(hash_code1[u as usize] & mask, min_hash2);
            }
            hash_code2[v as usize] = (min_hash << 16) + min_hash2;
        }
    } else {
        for v in 0..num_vertices {
            min_hash = u8::MAX as u32;
            min_hash2 = u8::MAX as u32;
            min_hash3 = u8::MAX as u32;
            min_hash4 = u8::MAX as u32;
            for &u in &reverse_graph[v as usize] {
                min_hash = min(hash_code1[u as usize] >> 24, min_hash);
                min_hash2 = min((hash_code1[u as usize] >> 16) & mask, min_hash2);
                min_hash3 = min((hash_code1[u as usize] >> 8) & mask, min_hash3);
                min_hash4 = min(hash_code1[u as usize] & mask, min_hash4);
            }
            hash_code2[v as usize] =
                (min_hash << 24) + (min_hash2 << 16) + (min_hash3 << 8) + min_hash4;
        }
    }

    std::mem::swap(&mut hash_code1, &mut hash_code2);

    let mut hash_id_pairs = vec![];
    for v in 0..num_vertices {
        hash_id_pairs.push((hash_code1[v as usize], reverse_graph[v as usize].len(), v));
    }

    hash_id_pairs.sort();
    hash_id_pairs.reverse();

    if in_bucket == 1 {
        let mut idx = 0;
        let mut next = 0;
        let mut j: u32;
        let mut unvisited = vec![false; num_vertices as usize];
        let mut qu = VecDeque::new();

        while idx < num_vertices {
            j = idx;
            while j < num_vertices && hash_id_pairs[idx as usize].0 == hash_id_pairs[j as usize].0 {
                unvisited[hash_id_pairs[j as usize].2 as usize] = true;
                j += 1;
            }

            for i in idx..j {
                let node = hash_id_pairs[i as usize].2 as usize;
                if unvisited[node] {
                    new_ids[node] = next;
                    next += 1;
                    unvisited[node] = false;

                    for &u in &graph[node] {
                        if u as usize == node {
                            continue;
                        }
                        qu.push_back(u);
                    }

                    while let Some(u) = qu.pop_front() {
                        if unvisited[u as usize] {
                            new_ids[u as usize] = next;
                            next += 1;
                            unvisited[u as usize] = false;

                            for &v in &graph[u as usize] {
                                if v == u {
                                    continue;
                                }
                                qu.push_back(v);
                            }
                        }
                    }
                }
            }

            idx = j;
        }
    } else if in_bucket == 2 {
        let mut idx = 0;
        let mut next = 0;
        let mut j: u32;
        let mut unvisited = vec![false; num_vertices as usize];
        let mut qu = BinaryHeap::new();

        while idx < num_vertices {
            j = idx;
            while j < num_vertices && hash_id_pairs[idx as usize].0 == hash_id_pairs[j as usize].0 {
                unvisited[hash_id_pairs[j as usize].2 as usize] = true;
                j += 1;
            }

            for i in idx..j {
                let node = hash_id_pairs[i as usize].2 as usize;
                if unvisited[node] {
                    new_ids[node] = next;
                    next += 1;
                    unvisited[node] = false;

                    for &u in &graph[node] {
                        if u as usize == node {
                            continue;
                        }
                        let nb = u;
                        if unvisited[nb as usize] {
                            qu.push((u32::MAX - next as u32, reverse_graph[nb as usize].len(), nb));
                            unvisited[nb as usize] = false;
                        }
                    }

                    while let Some((_, _, u)) = qu.pop() {
                        new_ids[u as usize] = next;
                        next += 1;

                        for &v in &graph[u as usize] {
                            if v == u {
                                continue;
                            }
                            let nb = v;
                            if unvisited[nb as usize] {
                                qu.push((u32::MAX - next as u32, reverse_graph[nb as usize].len(), nb));
                                unvisited[nb as usize] = false;
                            }
                        }
                    }
                }
            }

            idx = j;
        }
    }
    new_ids
}
