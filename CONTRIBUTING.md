# Contributing to orecchiette-fpv-drone-analog-rs

Local setup and the checks a change has to pass before a pull request.

## Quick start

```bash
git clone https://github.com/isaacbentley/orecchiette-fpv-drone-analog-rs.git
cd orecchiette-fpv-drone-analog-rs

cargo test --all-features
cargo clippy --all-features --all-targets -- -D warnings
cargo fmt --all --check
```

The crate declares `rust-version = "1.89"`, set by `wide` and
`safe_arch` in the dependency tree. It is a hard floor, not a
default-features-only one.

## What CI runs

Beyond the three commands above, a pull request also has to satisfy:

| Job | Command | Notes |
| :--- | :--- | :--- |
| Build & Test | `cargo test --all-features` | ubuntu, macOS and Windows |
| Clippy + Rustfmt | as above | Linux |
| cargo-deny | `cargo deny check` | licences and advisories, per `deny.toml` |
| cargo-hack | `cargo hack check --each-feature --no-dev-deps` | every feature in isolation |
| cargo-machete | `cargo machete` | unused dependencies |
| cargo-semver-checks | — | advisory only, does not block |
| Coverage | `cargo llvm-cov --all-features` | uploaded to Codecov; a drop does not fail the build |

`cargo hack` is the one most easily missed locally: a dependency used
under one feature but declared unconditionally passes a normal build and
fails there.

## Tests

Unit tests live inline in `src/` as `#[cfg(test)]` modules (113 of
them); `tests/` holds integration tests, and `benches/` a criterion
target (`cargo bench`).

Tests generate their own FM-modulated I/Q through `synthetic`, so the
repository carries no fixture files. Every module's tests share that
generator, so a generator change and a parser change cannot drift apart
independently.

For a change to DSP behaviour, generate a reference capture and replay
it rather than judging by eye alone:

```bash
cargo run --release --example make_reference_capture -- --standard pal
```

Those captures are correct by construction, so a decode failure against
one is a decoder fault. `fpv-viewer-rs` with `--debug` renders the
pipeline live when a visual check is also wanted.

## Code style

`rustfmt` defaults. Run `cargo fmt --all` before pushing.

Clippy runs with `-D warnings` in CI. Suppress a lint with an `// ALLOW:`
comment giving the reason.

Comments state what the code does and what was measured, with the
measurement's conditions. They do not narrate what the code used to do
or why an earlier version was wrong — `git log` holds that.

## Pull requests

- Commit messages: what changed, and why.
- Fill out the pull request template.

## License

By contributing, you agree your contributions will be licensed under
GPL-3.0-or-later, the same as the rest of the project.
