# Ising model simulator

## Description

This program is a simulator for an Ising model.  The simulator supports the following algorithms: Hill-climbing method, Metropolis method, Glauber dynamics and Stochastic cellular automata.

## Requirement

### Compiler

- GCC >= 5.5

### Font

This program use

- Consola font for windows,
- or Liberation Sans font for Linux.

You need to replace a default path `const std::string FontFile = ...` to valid one in `ising_model.h` because the font path depends on your operating system.

### Liblaries

- GLFW >= 3.0
- FTGL >= 2.1

## Usage

- **Start/Stop:** space key
- **Quit:** esc/q keys
- **Start/Stop cooling:** a key
- **Increase/Decrease temperature:** up/down keys
- **Change algorithm:** c key

## Getting Started

1. Download the source files.
1. If you have never installed any required libraries, then please install them.  When you use Debian-like linux distributions, for example, run a command:
    ```bash
    $ apt install libglfw3-dev libftgl-dev
    ````
1. Run `make clean all`.
1. If "Not found *font*" error occurs, then please modify the parameter `const std::string FontFile = ...` to the valid path in `ising_model.h`.

## Licence

This software is distributed under the terms of the MIT license reproduced [here](LICENSE).

## Author

Wandao123
