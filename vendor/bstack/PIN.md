# Vendored snapshot

Fuzz-only. Never a dependency of the published `bstack_raii` crate — only
`fuzz/Cargo.toml`'s `[patch.crates-io]` points at this directory.

- Source: `bstack` repo, branch `master`, commit `b848a421db6d5e23d98096727d5eec3be79faafb`
  ("Ignore Claude"), dated 2026-08-18.
- Reason: this commit carries the `debug-no-sync` feature (skips the real
  fsync/`F_FULLFSYNC` on every write, debug builds only) needed by
  [FUZZ.md](../../FUZZ.md)'s "Lever 2", ahead of its own crates.io release.
  `bstack`'s published `0.4.2` does not have it yet.
- `Cargo.toml`'s `version` field is left at `0.4.2` unchanged (matching the
  registry version it patches) — that's how `[patch.crates-io]` is meant to
  be used, not a claim this is the real 0.4.2 release contents.
- Taken via `git archive` of that exact commit (`Cargo.toml`, `LICENSE`,
  `README.md`, `CHANGELOG.md`, `src/`) — not a copy of any working tree, and
  in particular not of the sibling `bstack` checkout's in-progress merge on
  `resize`, which was mid-conflict when this was vendored.
- To refresh: re-run the same `git archive <new-commit> -- Cargo.toml LICENSE
  README.md CHANGELOG.md src | tar -x -C vendor/bstack` from a clean `bstack`
  commit and update the hash above.
