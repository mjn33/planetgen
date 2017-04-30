// Copyright (c) 2017 Matthew J. Nicholls
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#[derive(Clone, Copy, Debug)]
pub enum AllocError {
    /// An attempt was made to perform a zero-sized allocation, this is not
    /// supported.
    ZeroSizeAlloc,
    /// The allocation cannot be statisified due to being out of memory.
    OutOfMemory,
}

/// Represents a range in memory.
#[derive(Debug)]
pub struct AllocRange {
    start: usize,
    len: usize
}

impl AllocRange {
    /// Returns the start address of this range.
    pub fn start(&self) -> usize {
        self.start
    }
}

/// Simple allocator using best-fit.
pub struct SimpleAllocator {
    free_list: Vec<AllocRange>,
}

impl SimpleAllocator {
    /// Create a new `SimpleAllocator` managing a fixed `size` amount of memory.
    pub fn new(size: usize) -> SimpleAllocator {
        SimpleAllocator {
            free_list: vec![AllocRange { start: 0, len: size }],
        }
    }

    /// Try to allocate a block of memory of at least `len` size.
    pub fn alloc(&mut self, len: usize) -> Result<AllocRange, AllocError> {
        if len == 0 {
            return Err(AllocError::ZeroSizeAlloc);
        }
        // Find best fit
        let mut best_idx = None;
        let mut best_size = None;
        for (i, r) in self.free_list.iter().enumerate() {
            if r.len >= len && best_size.map_or(true, |s| s > r.len) {
                best_idx = Some(i);
                best_size = Some(r.len);
            }
        }

        let (best_idx, best_size) = match (best_idx, best_size) {
            (Some(best_idx), Some(best_size)) => (best_idx, best_size),
            _ => return Err(AllocError::OutOfMemory),
        };

        if best_size != len {
            let range = &mut self.free_list[best_idx];
            let rv = AllocRange {
                start: range.start,
                len: len,
            };
            range.start += len;
            range.len -= len;

            return Ok(rv);
        } else {
            return Ok(self.free_list.remove(best_idx));
        }
    }

    /// Free a block of memory, allowing it to be reused in subsequent
    /// allocations.
    pub fn free(&mut self, range: AllocRange) {
        let mut idx = None;
        for (i, r) in self.free_list.iter().enumerate() {
            if range.start < r.start {
                idx = Some(i);
                break;
            }
        }

        if let Some(idx) = idx {
            let merge_prev = {
                if idx > 0 {
                    let prev_range = &self.free_list[idx - 1];
                    prev_range.start + prev_range.len == range.start
                } else {
                    false
                }
            };

            let merge_next = {
                let next_range = &self.free_list[idx];
                next_range.start == range.start + range.len
            };

            if merge_next && merge_prev {
                {
                    let next_range_len = self.free_list[idx].len;
                    let prev_range = &mut self.free_list[idx - 1];
                    prev_range.len += range.len + next_range_len;
                }
                self.free_list.remove(idx);
            } else if merge_next {
                let next_range = &mut self.free_list[idx];
                next_range.start = range.start;
                next_range.len += range.len;
            } else if merge_prev {
                let prev_range = &mut self.free_list[idx - 1];
                prev_range.len += range.len;
            } else {
                self.free_list.insert(idx, range);
            }
        } else {
            if let Some(mut last) = self.free_list.pop() {
                if last.start + last.len == range.start {
                    last.len += range.len;
                    self.free_list.push(last);
                } else {
                    self.free_list.push(last);
                    self.free_list.push(range);
                }
            } else {
                self.free_list.push(range);
            }
        }
    }

    /// Returns the maximum possible allocation.
    pub fn max_alloc(&self) -> usize {
        self.free_list.iter().map(|r| r.len).max().unwrap_or(0)
    }

    /// Returns the amount of free memory, note this may be different
    /// to the value returned by `max_alloc()` due to external
    /// fragmentation.
    #[allow(dead_code)]
    pub fn free_mem(&self) -> usize {
        self.free_list.iter().map(|r| r.len).sum()
    }

    /// Calculates the external fragmentation, which is a value between 0.0 and
    /// 1.0, high external fragmentation means there is a lot of free memory but
    /// only a small fraction of that can be used in a single allocation.
    #[allow(dead_code)]
    pub fn external_fragmentation(&self) -> f32 {
        if self.free_mem() != 0 {
            1.0 - self.max_alloc() as f32 / self.free_mem() as f32
        } else {
            0.0
        }
    }
}
