# Contributing to orecchiette-fpv-drone-analog-rs

First off, thank you for considering contributing to `orecchiette-fpv-drone-analog-rs`! It's people like you that make the open-source SDR community thrive. This document explains how to set up your environment and verify your changes locally before opening a pull request.

## Quick Start

```bash
git clone https://github.com/isaacbentley/orecchiette-fpv-drone-analog-rs.git
cd orecchiette-fpv-drone-analog-rs

# Run the standard validation suite
cargo test
cargo clippy --all-features --all-targets -- -D warnings
cargo fmt --all --check
```

## Testing Your DSP Changes

This crate contains the pure DSP algorithms for analog FPV demodulation and temporal noise reduction. 
If you are tweaking algorithmic parameters or adding new decoding modes:
- Please ensure that unit tests under `tests/` pass.
- Since DSP changes can be subjective (e.g. video quality), we strongly recommend testing your branch visually using `fpv-viewer-rs` with offline `.sigmf` files to confirm noise and tracking stability aren't regressions.

## Code Style

We use standard `rustfmt` defaults. Please run `cargo fmt --all` before pushing.

Clippy is run with `-D warnings` in CI. If a lint is genuinely wrong for the situation, allow it with a `// ALLOW:` justification comment explaining why.

## Pull Requests

- **Commit messages:** Describe *why* the change is needed and *what* it changes.
- **Templates:** Please fill out the Pull Request template when opening a PR. Checkboxes are provided for CI validations.

## License

By contributing, you agree your contributions will be licensed under GPL-3.0-or-later, the same as the rest of the project.
