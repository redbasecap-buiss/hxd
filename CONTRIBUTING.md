# Contributing to FLUX

Thank you for your interest in contributing to FLUX! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/flux.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run tests: `cargo test`
6. Format: `cargo fmt`
7. Lint: `cargo clippy -- -D warnings`
8. Commit and push
9. Open a Pull Request

## Code Style

- Follow standard Rust conventions (`rustfmt` defaults)
- Document all public APIs with `///` doc comments
- Include unit tests for new functionality
- Keep functions focused and small

## Commit Messages

Use conventional commits:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `perf:` performance improvements
- `test:` test additions/changes
- `refactor:` code refactoring

## Testing

- Unit tests: `cargo test`
- Validation tests: `cargo run --release -- validate`
- Benchmarks: `cargo run --release -- bench`

## Reporting Issues

Please include:
- FLUX version (`cargo run -- --version`)
- Operating system and hardware
- Minimal reproduction case
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
