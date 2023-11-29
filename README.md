[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
![build workflow](https://github.com/alexdesander/chikage/actions/workflows/build_and_test.yml/badge.svg)

# chikage
Simple, easy to understand and barebones math library for game and graphics development.

<p align="center">
  <img src="./chikage_logo.png" width="25%"/>
</p>

## Features

- 2-4D floating point vectors
- Square floating point matrices of order 2-4
- 3D floating point rotors (understandable quaternions)

## Goals

- Simple and easy to understand code
  - No macros
  - No simd
  - No obscure optimizations
  - Documented functionality
- Barebones
- Comprehensive unit tests
- No dependencies (exception: std)
- Provide few buildings but all building blocks. Examples:
  - Chikage provides 4D square matrices, but no projection matrix.
  - Chikage provides multivectors, but also rotors (reasoning: optimization and simplicity for the end user) (NOT YET IMPLEMENTED).

## Planned features
- ✅ 2-4D floating point vectors
- ✅ floating point square matrices of order 2-4
- ⬜️ 2-4D floating point rotors
    - ✔ 3D rotor
    - ✖ 2D rotor
    - ✖ 4D rotor
- ⬜️ More geometric algebra in 2-4D (details to be decided)
- (Featureset is open ended)