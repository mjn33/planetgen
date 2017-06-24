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

//! TODO: this module needs more (some!) documentation

use std::any::TypeId;
use std::marker::PhantomData;

pub trait Container {
    type Item;
    type HandleType: Copy;

    fn push(&mut self, v: Self::Item);
    fn swap_remove(&mut self, idx: usize);
}

#[derive(Clone, Copy, Debug)]
pub struct GenericHandle {
    type_id: TypeId,
    id: u32,
    gen: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Handle<T: Copy + 'static> {
    id: u32,
    gen: u32,
    phantom: PhantomData<T>,
}

impl<T: Copy> PartialEq for Handle<T> {
    fn eq(&self, other: &Handle<T>) -> bool {
        self.id == other.id && self.gen == other.gen
    }
}

impl<T: Copy> Handle<T> {
    pub fn from_generic_handle(handle: GenericHandle) -> Result<Handle<T>, ()> {
        let type_id = TypeId::of::<Self>();
        if handle.type_id != type_id {
            return Err(());
        } else {
            return Ok(Handle {
                id: handle.id,
                gen: handle.gen,
                phantom: PhantomData,
            });
        }
    }

    pub fn into_generic_handle(self) -> GenericHandle {
        let type_id = TypeId::of::<Self>();
        GenericHandle {
            type_id,
            id: self.id,
            gen: self.gen,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct HandleData {
    gen: u32,
    data_idx: u32,
}

#[derive(Debug)]
pub struct ObjectManager<T: Container> {
    handle_tbl: Vec<HandleData>,
    data_tbl: Vec<u32>,
    free_list: Vec<u32>,
    pub c: T,
}

impl<T: Container> ObjectManager<T> {
    pub fn new(c: T) -> ObjectManager<T> {
        ObjectManager {
            handle_tbl: Vec::new(),
            data_tbl: Vec::new(),
            free_list: Vec::new(),
            c
        }
    }

    pub fn remove(&mut self, handle: Handle<T::HandleType>) -> Result<(), ()> {
        let HandleData { gen, data_idx } = self.handle_tbl[handle.id as usize];
        if gen != handle.gen {
            return Err(());
        }
        self.remove_internal(handle.id, data_idx as u32);
        Ok(())
    }

    pub fn remove_idx(&mut self, data_idx: usize) {
        let id = self.data_tbl[data_idx];
        self.remove_internal(id, data_idx as u32);
    }

    fn remove_internal(&mut self, id: u32, data_idx: u32) {
        self.handle_tbl[id as usize].gen += 1;
        self.data_tbl.swap_remove(data_idx as usize);
        self.c.swap_remove(data_idx as usize);

        if let Some(&v) = self.data_tbl.get(data_idx as usize) {
            self.handle_tbl[v as usize].data_idx = data_idx;
        }
        self.free_list.push(id);
    }

    pub fn add(&mut self, value: T::Item) -> Handle<T::HandleType> {
        if let Some(id) = self.free_list.pop() {
            let data_idx = self.data_tbl.len() as u32;
            let gen = self.handle_tbl[id as usize].gen;
            self.handle_tbl[id as usize].data_idx = data_idx;
            self.data_tbl.push(id);
            self.c.push(value);

            Handle {
                id,
                gen,
                phantom: PhantomData,
            }
        } else {
            let data_idx = self.data_tbl.len() as u32;
            let gen = 0;
            let id = self.handle_tbl.len() as u32;
            self.handle_tbl.push(HandleData { gen, data_idx });
            self.data_tbl.push(id);
            self.c.push(value);

            Handle {
                id,
                gen,
                phantom: PhantomData,
            }
        }
    }

    pub fn handle(&self, data_idx: usize) -> Handle<T::HandleType> {
        let id = self.data_tbl[data_idx];
        let HandleData { gen, .. } = self.handle_tbl[id as usize];
        Handle {
            id,
            gen,
            phantom: PhantomData,
        }
    }

    pub fn data_idx_checked(&self, handle: Handle<T::HandleType>) -> Result<usize, ()> {
        let HandleData { gen, data_idx } = self.handle_tbl[handle.id as usize];
        if gen != handle.gen {
            Err(())
        } else {
            Ok(data_idx as usize)
        }
    }

    pub fn is_handle_valid(&self, handle: Handle<T::HandleType>) -> bool {
        let HandleData { gen, .. } = self.handle_tbl[handle.id as usize];
        gen == handle.gen
    }
}

#[cfg(test)]
mod test {
    use super::*;

    struct FloatContainer {
        floats: Vec<f32>,
    }

    #[derive(Copy, Clone)]
    struct Float;

    impl Container for FloatContainer {
        type Item = f32;
        type HandleType = Float;

        fn push(&mut self, value: Self::Item) {
            self.floats.push(value);
        }

        fn swap_remove(&mut self, idx: usize) {
            self.floats.swap_remove(idx);
        }
    }


    #[test]
    fn test_manager() {
        let mut test = ObjectManager::new(FloatContainer { floats: Vec::new() });
        let v1 = 33.4;
        let h1 = test.add(v1);
        let v2 = 44.5;
        let h2 = test.add(v2);
        let v3 = 55.6;
        let h3 = test.add(v3);

        assert_eq!(test.c.floats[test.data_idx_checked(h1).unwrap() as usize], v1);
        assert_eq!(test.c.floats[test.data_idx_checked(h2).unwrap() as usize], v2);
        assert_eq!(test.c.floats[test.data_idx_checked(h3).unwrap() as usize], v3);

        test.remove(h2).unwrap();

        assert!(test.data_idx_checked(h2).is_err());
        assert!(test.remove(h2).is_err());

        let v4 = 66.7;
        let h4 = test.add(v4);

        assert_eq!(test.c.floats[test.data_idx_checked(h4).unwrap() as usize], v4);

        test.remove(h1).unwrap();
        test.remove(h3).unwrap();
        test.remove(h4).unwrap();

        // TODO:
        // - double-free
        // - use-after-free
        // - check invariants
    }
}
