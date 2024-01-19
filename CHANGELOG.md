# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

* Don't use virtual inheritance to implement `finite_difference_wrapper`
* Build on Windows
* Fix Armijo lineseach exit condition

### Changed

* Changed the minimum version of eigenpy to 3.2.0

## [0.3.1] - 2023-12-21

### Fixed

* Fixed a missing `#pragma once` in `linalg/bunchkaufman.hpp`
* Fix benchmarks build with MacOS
* Fix pkg-config definitions

## [0.3.0] - 2023-12-19

### Added

* This is the first release of `proxsuite-nlp`. This library is a joint effort between LAAS-CNRS and INRIA, and will be maintained and expanded in the future. Please provide constructive feedback and contribute!

[Unreleased]: https://github.com/Simple-Robotics/proxsuite-nlp/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/Simple-Robotics/proxsuite-nlp/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Simple-Robotics/proxsuite-nlp/releases/tag/v0.3.0
