# PlanetGen

A procedurally generated planetary surface using quad-trees.

## Controls

| Key    | Description            |
|--------|------------------------|
| W      | Move forward           |
| S      | Move backward          |
| A      | Move left              |
| D      | Move right             |
| Q      | Roll counter-clockwise |
| E      | Roll clockwise         |
| LShift | Move faster            |
| ,      | Slow down time         |
| .      | Speed up time          |

## Building

To build PlanetGen you will need the SDL2 libraries; you will also need a
nightly version of the Rust compiler and Cargo. To make a release build run:

```
$ cargo build --release
```

### Linux

You can get the SDL2 libraries from your distribution's repositories; you will
need the development versions of these packages as well.

### Windows

On Windows you will either need to get the libraries from http://www.libsdl.org/
or build them yourself. SDL2 library and DLL files should be put under
`lib/$target/` and `bin/$target/` directories respectively where `$target` is
the target triple as understood by Rust. For example, use
`x86_64-pc-windows-msvc` as the target when building 64-bit binaries using the
Microsoft Visual C toolchain.

## Licence

PlanetGen is licensed under the MIT licence (see [LICENSE-MIT](LICENSE-MIT)),
portions are licensed under the GNU LGPL (rust-libnoise, see
[LICENSE-LGPL](LICENSE-LGPL)). See [COPYRIGHT](COPYRIGHT) for more information.
