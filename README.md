# Rendering Demo

[![CI Status](https://github.com/NMGardiner/rendering-demo/actions/workflows/generate-docs.yml/badge.svg)](https://github.com/NMGardiner/rendering-demo/actions)

## Table of Contents

1. [Features](#features)
1. [Dependencies](#dependencies)
1. [Building](#building)
1. [Running](#running)
1. [License](#license)

## Features

- Supports both Linux and Windows.

## Dependencies

Building this project requires a device and driver supporting Vulkan 1.3, and the Vulkan SDK if you wish to enable validation layers. 

## Building

Execute the following in the project root:

`cargo build --release`

## Running

If you're building the project manually, execute the following in the project root:

`cargo run --release`

If you instead downloaded a release binary, execute `demo-app.exe`.

In both cases, this will run the demo scene.

## License

TODO