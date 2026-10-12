# Vendoring patches

One patch per vendored package, holding everything the copy under `deps/<package>` has that the
package's `integration` branch does not. `deps/vendor.jl sync` re-applies the patch after every
`git subtree pull`, so the pair (`integration` branch, patch) reproduces the vendored tree
exactly -- which is what makes a re-vendor a mechanical operation instead of a three-way merge.

Regenerate with:

```
julia deps/vendor.jl rebuild    # assemble each fork's `integration` branch from deps/vendor.toml
julia deps/vendor.jl patch      # diff each integration branch against deps/<package>
```

`patch` writes the diff of a clean `git archive` of `integration` against the working `deps/`
tree, so it is always exactly the current delta: nothing about it is hand-maintained.

## What belongs in them

Only what vendoring itself forces. Today that is:

- **Relative imports.** A submodule cannot `using AbstractOperators`; it has to say
  `using ..AbstractOperators`.
- **The inlined extension.** An `ext/` directory never loads for a submodule, so
  `ProximalOperators`' `RecursiveArrayToolsExt` is included from `src/` by hand.
- **OSQP, removed.** Ristretto does not use it, so neither the weak dependency nor `IndPolyhedral`,
  whose only implementation needs it, is vendored.
- **The vendored `[sources]` paths**, pointing at the sibling copies under `deps/` rather than at
  a developer's own checkouts.

Nothing else. Three destinations exist for a change found while developing Ristretto, and the patch is
none of them:

- A fix or feature that would make sense to the upstream package goes on the branch that owns
  that code -- as another commit on the branch whose PR introduced it, not a new branch -- and
  reaches `deps/` through `integration` on the next sync.
- A change with no owning branch gets one new branch for that one idea, added to
  `deps/vendor.toml`.
- A change that is really about MRI rather than about the dependency belongs in Ristretto's own `src/`.

`patch` regenerates these files from the current trees, so a hunk that is none of the above
appears the moment someone edits `deps/` by hand -- which is what makes the rule enforceable
rather than merely stated. A patch that stops applying after a rebuild is good news: whatever it
carried is in the branch now.

## The integration branches these are relative to

`integration` is disposable and rebuilt from `deps/vendor.toml` on demand. Some of its merges
conflict -- mostly where a fork branch is still based on an older `master` rather than on the
branch the manifest says it is stacked on. Those conflicts were resolved once by taking the file
as it stands in `deps/`, on the grounds that the vendored tree is the version that is known to
work, and `git rerere` is enabled in each dependency checkout so the resolutions replay
automatically on the next rebuild. Resolving a conflict this way is a stopgap: the real fix is to
rebase the branch onto its manifest parent, after which the conflict does not arise.
