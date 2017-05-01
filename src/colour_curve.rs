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

#[derive(Clone, Default)]
pub struct ColourCurve {
    control_points: Vec<ControlPoint>,
}

#[derive(Copy, Clone)]
struct ControlPoint {
    input_value: f64,
    output_value: (u8, u8, u8, u8),
}

fn clamp<T: Ord>(value: T, lower_bound: T, upper_bound: T) -> T {
    if value < lower_bound {
        lower_bound
    } else if value > upper_bound {
        upper_bound
    } else {
        value
    }
}

impl ColourCurve {
    pub fn new() -> ColourCurve {
        Default::default()
    }

    pub fn add_control_point(&mut self, input_value: f64, output_value: (u8, u8, u8, u8)) {
        if input_value.is_nan() {
            // With this check the `unwrap()` in the binary search should always
            // succeed.
            panic!("Tried to insert NaN input_value!");
        }
        let f = |x: &ControlPoint| x.input_value.partial_cmp(&input_value).unwrap();
        match self.control_points.binary_search_by(f) {
            Ok(_) => {
                panic!("Control point with given input value already exists!");
            }
            Err(idx) => {
                self.control_points
                    .insert(idx,
                            ControlPoint {
                                input_value: input_value,
                                output_value: output_value,
                            });
            }
        }
    }

    pub fn get_colour(&self, value: f64) -> (u8, u8, u8, u8) {
        let f = |x: &ControlPoint| x.input_value.partial_cmp(&value).unwrap();
        let idx_pos = match self.control_points.binary_search_by(f) {
            Ok(idx) => idx as isize + 1,
            Err(idx) => idx as isize,
        };

        let idx0 = clamp(idx_pos - 1, 0, self.control_points.len() as isize - 1) as usize;
        let idx1 = clamp(idx_pos, 0, self.control_points.len() as isize - 1) as usize;

        if idx0 == idx1 {
            return self.control_points[idx0].output_value;
        }

        let input0 = self.control_points[idx0].input_value;
        let input1 = self.control_points[idx1].input_value;
        let alpha = (value - input0) / (input1 - input0);

        let (r0, g0, b0, a0) = self.control_points[idx0].output_value;
        let (r1, g1, b1, a1) = self.control_points[idx1].output_value;

        let r = (((1.0 - alpha) * r0 as f64) + (alpha * r1 as f64)) as u8;
        let g = (((1.0 - alpha) * g0 as f64) + (alpha * g1 as f64)) as u8;
        let b = (((1.0 - alpha) * b0 as f64) + (alpha * b1 as f64)) as u8;
        let a = (((1.0 - alpha) * a0 as f64) + (alpha * a1 as f64)) as u8;

        (r, g, b, a)
    }
}
