#!/usr/bin/env julia

# Tooling for the vendored dependency stack declared in `deps/vendor.toml`.
#
#     julia deps/vendor.jl check [pkg...]        compare the manifest with GitHub and git
#     julia deps/vendor.jl rebuild [pkg...]      rebuild each fork's `integration` branch
#     julia deps/vendor.jl sync [pkg...]         project `integration` into `deps/`
#     julia deps/vendor.jl status [--json]       the state of the whole stack
#
# With no package names, every package in the manifest is processed. Only the standard library is
# used, so the script runs with a bare `julia` and no project environment. `check` and `status`
# read GitHub through the `gh` CLI; if `gh` is missing or unauthenticated, they degrade to what
# git alone can tell and say so.
#
# Where each fork is checked out is machine state, not a property of the stack, so it lives in
# `deps/vendor.local.toml` -- untracked, one `<Package> = "<path>"` line per checkout that exists
# on this machine. A package with no entry there is simply not checked out here: `check` still
# reports everything GitHub can answer for it, and `rebuild`/`patch`/`sync`, which need a working
# tree, say which entry is missing instead of failing on an empty path.

module VendorTool

using Dates
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const MANIFEST = joinpath(ROOT, "deps", "vendor.toml")
const LOCAL_MANIFEST = joinpath(ROOT, "deps", "vendor.local.toml")

struct Branch
    name::String
    parent::String
    pr::String
end

struct Package
    name::String
    fork::String
    upstream::String
    base::String
    prefix::String
    path::String
    prune::Vector{String}
    stack::Vector{Branch}
end

# ---------------------------------------------------------------------------------------------
# manifest

"""
Paths of the local checkouts, read from the untracked `deps/vendor.local.toml`. Missing file, or
a package with no entry in it, simply means "not checked out on this machine"; the empty path is
what `has_checkout` tests.
"""
function load_local_paths()
    isfile(LOCAL_MANIFEST) || return Dict{String, String}()
    raw = TOML.parsefile(LOCAL_MANIFEST)
    paths = Dict{String, String}()
    for (name, value) in raw
        value isa AbstractString || continue
        paths[name] = expanduser(value)
    end
    return paths
end

has_checkout(pkg::Package) = !isempty(pkg.path) && isdir(joinpath(pkg.path, ".git"))

"""
Path of `pkg`'s working checkout, or an error naming what to add to `deps/vendor.local.toml`.
Every command that needs a working tree goes through this.
"""
function require_checkout(pkg::Package)
    has_checkout(pkg) && return pkg.path
    error(
        isempty(pkg.path) ?
            "$(pkg.name) is not checked out on this machine: add `$(pkg.name) = \"/path/to/checkout\"` " *
            "to $(relpath(LOCAL_MANIFEST, ROOT)) (untracked) and clone $(pkg.fork) there." :
            "$(pkg.name): `$(pkg.path)` (from $(relpath(LOCAL_MANIFEST, ROOT))) is not a git checkout",
    )
end

function load_manifest()
    raw = TOML.parsefile(MANIFEST)
    local_paths = load_local_paths()
    packages = Package[]
    for (name, entry) in sort(collect(raw); by = first)
        stack = [
            Branch(b["branch"], get(b, "parent", ""), get(b, "pr", "")) for
            b in get(entry, "stack", Dict{String,Any}[])
        ]
        push!(
            packages,
            Package(
                name,
                entry["fork"],
                entry["upstream"],
                entry["base"],
                entry["prefix"],
                get(local_paths, name, ""),
                String.(get(entry, "prune", String[])),
                stack,
            ),
        )
    end
    return packages
end

select(packages, names) =
    isempty(names) ? packages : filter(p -> p.name in names, packages)

"""
Delete everything the package's `prune` globs match inside `dir`. Ristretto ships only the code it
loads: a vendored package's own test suite, documentation, benchmarks and CI never run here (they
run in the fork, against the real package), and an `ext/` directory cannot load at all for a
package included as a submodule. Pruning them keeps `deps/` to what is actually compiled, and
keeps the difference out of `deps/patches/`, where it would otherwise be thousands of lines of
deletions with nothing to say.
"""
function prune!(dir::AbstractString, globs::Vector{String})
    isempty(globs) && return
    run(`bash -c $("shopt -s nullglob dotglob; cd " * dir * " && rm -rf -- " * join(globs, " "))`)
    return
end

"""
Validate that the stack is a forest whose parents are declared before their children, so that
merging the branches in file order always merges a parent before the branch stacked on it.
"""
function validate(pkg::Package)
    problems = String[]
    seen = Set{String}()
    for b in pkg.stack
        b.name in seen && push!(problems, "$(pkg.name): branch `$(b.name)` declared twice")
        if !isempty(b.parent) && !(b.parent in seen)
            push!(
                problems,
                "$(pkg.name): `$(b.name)` is stacked on `$(b.parent)`, which is not declared " *
                "before it",
            )
        end
        push!(seen, b.name)
    end
    return problems
end

# ---------------------------------------------------------------------------------------------
# shelling out

struct CommandFailure <: Exception
    cmd::Cmd
    output::String
end

Base.showerror(io::IO, e::CommandFailure) = print(io, "failed: ", e.cmd, "\n", e.output)

"""
Run `cmd`, returning its trimmed output. Returns `nothing` instead of throwing when
`nothing_on_error` is set, which is how the optional `gh` queries stay optional.
"""
function capture(cmd::Cmd; nothing_on_error::Bool = false)
    out = IOBuffer()
    try
        run(pipeline(cmd; stdout = out, stderr = out))
    catch
        nothing_on_error && return nothing
        throw(CommandFailure(cmd, String(take!(out))))
    end
    return strip(String(take!(out)))
end

git(pkg::Package, args...) = capture(`git -C $(pkg.path) $(collect(args))`)
git_or_nothing(pkg::Package, args...) =
    capture(`git -C $(pkg.path) $(collect(args))`; nothing_on_error = true)
here(args...) = capture(`git -C $ROOT $(collect(args))`)

const GH_AVAILABLE = Ref{Union{Nothing,Bool}}(nothing)

function gh_available()
    if GH_AVAILABLE[] === nothing
        GH_AVAILABLE[] =
            Sys.which("gh") !== nothing &&
            capture(`gh auth status`; nothing_on_error = true) !== nothing
    end
    return GH_AVAILABLE[]::Bool
end

"""
State of one pull request as GitHub sees it: `(state, base, url)`, or `nothing` when the PR is
unknown or `gh` is unavailable. `ref` is the `owner/repo#number` form used in the manifest.
"""
function pr_state(ref::AbstractString)
    gh_available() || return nothing
    (isempty(ref) || !occursin('#', ref)) && return nothing
    repo, number = split(ref, '#')
    out = capture(
        `gh pr view $number --repo $repo --json state,baseRefName,url,isDraft`;
        nothing_on_error = true,
    )
    out === nothing && return nothing
    # `gh` returns JSON; pick out the few fields we need without a JSON dependency.
    state = match(r"\"state\":\"([^\"]*)\"", out)
    base = match(r"\"baseRefName\":\"([^\"]*)\"", out)
    url = match(r"\"url\":\"([^\"]*)\"", out)
    draft = occursin("\"isDraft\":true", out)
    (state === nothing || base === nothing) && return nothing
    return (
        state = state.captures[1],
        base = base.captures[1],
        url = url === nothing ? "" : url.captures[1],
        draft = draft,
    )
end

# ---------------------------------------------------------------------------------------------
# check

"""
`owner/repo` of a fork or upstream URL, as `gh` wants it.
"""
function repo_slug(url::AbstractString)
    m = match(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    return m === nothing ? "" : "$(m.captures[1])/$(m.captures[2])"
end

"""
Names of the branches the fork actually has on GitHub, or `nothing` when `gh` cannot say.
Queried from GitHub rather than from remote-tracking refs, so the answer does not depend on when
the local checkout last fetched -- or on there being a local checkout at all.
"""
function fork_branches(pkg::Package)
    gh_available() || return nothing
    slug = repo_slug(pkg.fork)
    isempty(slug) && return nothing
    out = capture(
        `gh api --paginate -q ".[].name" repos/$slug/branches`;
        nothing_on_error = true,
    )
    return out === nothing ? nothing : filter(!isempty, split(out, '\n'))
end

# Absolute paths of one machine, and the scratch worktrees an agent leaves behind. A fork branch
# is public code: it may name a sibling checkout the way upstream does, but never a path that
# exists only here. Ristretto's own `[sources]` rewriting belongs in `deps/patches/<package>.patch`.
const LOCAL_PATH_PATTERN = raw"(/project/|/home/|/scratch/|\.claude/worktrees/)"

"""
Lines of `origin/<branch>` that hardcode a path of this machine, as `file:line:text`, or
`nothing` when the search could not run. Needs a checkout; without one the scan is skipped and
`check` says so.

Lines the upstream base already has are not reported: `benchmarks/` in more than one of these
packages carries an absolute path of whoever wrote it years ago, which is upstream's business and
not something a fork branch introduced.
"""
function local_path_hits(pkg::Package, branch::AbstractString)
    mine = _path_grep(pkg, "origin/$branch")
    mine === nothing && return nothing
    theirs = _path_grep(pkg, pkg.base)
    theirs === nothing && return mine
    inherited = Set(_hit_text.(theirs))
    return filter(h -> !(_hit_text(h) in inherited), mine)
end

# `file:line:text` -> `text`, so a line can be recognised across revisions that moved it.
_hit_text(hit::AbstractString) = let parts = split(hit, ':'; limit = 3)
    length(parts) == 3 ? strip(parts[3]) : strip(hit)
end

function _path_grep(pkg::Package, rev::AbstractString)
    # `-c grep.threads=1`: on a busy login node `git grep`'s worker threads hit the per-user
    # process limit and it dies with `failed to create thread: Resource temporarily unavailable`
    # -- which exits non-zero and would otherwise be indistinguishable from "nothing matched",
    # i.e. would report a contaminated branch as clean.
    cmd = `git -C $(pkg.path) -c grep.threads=1 grep -n -E $LOCAL_PATH_PATTERN $rev
        -- "*.toml" "*.jl" "*.yml" "*.yaml"`
    # `git grep` exits non-zero both when nothing matched (the clean case, no output) and when it
    # could not search at all, so the output has to be read either way: a `fatal:` that reported
    # itself as "no hits" would be a false all-clear.
    out = try
        capture(cmd)
    catch e
        e isa CommandFailure || rethrow()
        strip(e.output)
    end
    occursin("fatal:", out) && return nothing
    return [replace(l, "$rev:" => "") for l in split(out, '\n') if !isempty(l)]
end

"""
Report every way the stack disagrees with itself: a branch whose PR is based on something other
than its manifest parent, a branch with no PR at all, a branch whose PR has already merged (its
code should come from `base` instead), a branch the fork does not have, a branch whose local tip
is ahead of the fork's, a branch that hardcodes a path of this machine, and a branch on the fork
that no manifest entry refers to.

Branch existence comes from GitHub (`gh`), not from remote-tracking refs, so a stale local fetch
cannot make an unpushed branch look pushed. Everything that needs file content or local commits
additionally needs a checkout listed in `deps/vendor.local.toml`; `origin` and `upstream` are
fetched first for those, unless `fetch` is false.
"""
function check(packages; fetch::Bool = true)
    findings = 0
    for pkg in packages
        println("== ", pkg.name)
        checkout = has_checkout(pkg)
        if !checkout
            println(
                "  note: not checked out here (no entry in $(relpath(LOCAL_MANIFEST, ROOT))); " *
                "only what GitHub can answer is checked",
            )
        elseif fetch
            for remote in ("origin", "upstream")
                git_or_nothing(pkg, "fetch", "--prune", remote) === nothing &&
                    println("  note: `git fetch $remote` failed in $(pkg.path)")
            end
        end
        for problem in validate(pkg)
            println("  manifest: ", problem)
            findings += 1
        end
        remote_branches = fork_branches(pkg)
        for b in pkg.stack
            pushed = remote_branches === nothing ?
                (checkout && git_or_nothing(pkg, "rev-parse", "--verify", "--quiet", "origin/$(b.name)") !== nothing) :
                (b.name in remote_branches)
            if !pushed
                println("  $(b.name): not pushed -- the fork has no such branch")
                findings += 1
            elseif checkout
                local_tip = git_or_nothing(pkg, "rev-parse", "--verify", "--quiet", b.name)
                remote_tip = git_or_nothing(pkg, "rev-parse", "--verify", "--quiet", "origin/$(b.name)")
                if local_tip !== nothing && remote_tip !== nothing && local_tip != remote_tip
                    counts = git(pkg, "rev-list", "--left-right", "--count", "origin/$(b.name)...$(b.name)")
                    behind, ahead = split(counts)
                    if ahead != "0"
                        println(
                            "  $(b.name): $(ahead) local commit(s) not pushed to the fork" *
                            (behind == "0" ? "" : " (and $(behind) on the fork not merged locally)"),
                        )
                        findings += 1
                    end
                end
            end
            if pushed && checkout
                hits = local_path_hits(pkg, b.name)
                if hits === nothing
                    println("  $(b.name): could not scan for local paths (`git grep` failed)")
                    findings += 1
                elseif !isempty(hits)
                    println("  $(b.name): $(length(hits)) pushed line(s) hardcode a local path:")
                    for h in first(hits, 5)
                        println("      ", h)
                    end
                    length(hits) > 5 && println("      ... and $(length(hits) - 5) more")
                    findings += 1
                end
            end
            if isempty(b.pr)
                println("  $(b.name): vendored but no PR anywhere -- untracked work")
                findings += 1
                continue
            end
            info = pr_state(b.pr)
            info === nothing && continue
            if info.state == "MERGED"
                println(
                    "  $(b.name): $(b.pr) is merged -- drop the entry and take the code from " *
                    "$(pkg.base)",
                )
                findings += 1
            elseif info.state == "CLOSED"
                println("  $(b.name): $(b.pr) is closed, but the branch is still vendored")
                findings += 1
            end
            # The PR base is the claim GitHub makes about the stack; the manifest parent is the
            # claim this repository makes. A mismatch means the PR will conflict when its
            # intended parent lands.
            intended = isempty(b.parent) ? last(split(pkg.base, '/')) : b.parent
            if info.state == "OPEN" && info.base != intended
                println(
                    "  $(b.name): $(b.pr) is based on `$(info.base)`, the stack wants " *
                    "`$(intended)`",
                )
                findings += 1
            end
        end
        # A branch on the fork that no entry names is work that will never reach `deps/`: either
        # it is finished and belongs in the manifest, or it is dead and belongs deleted. The
        # fork's own copy of the upstream default branch, the generated `integration` branch and
        # the documentation deployed from it (`gh-pages`) are not stack entries and are expected
        # to be there.
        if remote_branches !== nothing
            declared = Set(b.name for b in pkg.stack)
            expected = Set(["integration", "gh-pages", "master", "main", last(split(pkg.base, '/'))])
            extra = sort([b for b in remote_branches if !(b in declared) && !(b in expected)])
            if !isempty(extra)
                println("  fork branches no manifest entry refers to: ", join(extra, ", "))
                findings += 1
            end
        end
    end
    gh_available() ||
        println("\nnote: `gh` unavailable, so PR state, base and the fork's branch list were not checked")
    println("\n", findings == 0 ? "stack is consistent" : "$findings finding(s)")
    return findings
end

# ---------------------------------------------------------------------------------------------
# rebuild

"""
Scratch worktree in which a package's `integration` branch is built. A worktree rather than the
package's own checkout, so that rebuilding never disturbs whatever branch is checked out there
and never touches uncommitted work in it.
"""
worktree(pkg::Package) = joinpath(ROOT, ".vendor-work", pkg.name)

"""
Resolve a manifest branch to something git can merge: the fork's copy when it has one, otherwise
a local branch of the same name. A branch that exists only locally is reported by `check`; it is
still merged here, because `deps/` already contains its code.
"""
function resolve(pkg::Package, branch::AbstractString)
    for ref in ("origin/$branch", branch)
        git_or_nothing(pkg, "rev-parse", "--verify", "--quiet", ref) === nothing || return ref
    end
    error("$(pkg.name): neither `origin/$branch` nor `$branch` exists in $(pkg.path)")
end

"""
Rebuild `integration` for each package: reset it to `base`, then merge every manifest branch in
file order. Because a parent is always declared before the branches stacked on it, the merge
order is the order the PRs are meant to land in, and the result is one ref meaning "everything I
have" for that package.

A merge conflict stops that package and leaves the worktree as it is, so the conflict can be
resolved by hand; the resolution belongs on the branch that caused it, not in the worktree.
"""
function rebuild(packages; push::Bool = true, fetch::Bool = true)
    failed = String[]
    for pkg in packages
        problems = validate(pkg)
        isempty(problems) || error(join(problems, "\n"))
        require_checkout(pkg)
        println("== ", pkg.name, " (", pkg.path, ")")
        if fetch
            git(pkg, "fetch", "--prune", "origin")
            git(pkg, "fetch", "--prune", "upstream")
        end
        wt = worktree(pkg)
        ispath(wt) && capture(
            `git -C $(pkg.path) worktree remove --force $wt`;
            nothing_on_error = true,
        )
        # A worktree directory deleted by hand leaves its registration behind, and that stale
        # entry still counts as `integration` being checked out somewhere.
        git(pkg, "worktree", "prune")
        mkpath(dirname(wt))
        git(pkg, "worktree", "add", "--force", "-B", "integration", wt, pkg.base)
        conflict = nothing
        for b in pkg.stack
            ref = resolve(pkg, b.name)
            print("  merge ", rpad(b.name, 36), " (", ref, ")")
            # `rerere` makes a conflict resolution a one-off: resolved by hand once in this
            # worktree, replayed automatically on every later rebuild. Without it the same
            # resolutions would have to be redone each time, since `integration` is disposable.
            out = capture(
                `git -C $wt -c rerere.enabled=true -c rerere.autoupdate=true
                    merge --no-ff -m "integration: $(b.name)" $ref`;
                nothing_on_error = true,
            )
            if out === nothing
                # `rerere` stages a replayed resolution but never commits it, so a merge that
                # leaves nothing unresolved is a success that only looks like a failure.
                unresolved = capture(`git -C $wt diff --name-only --diff-filter=U`)
                if isempty(unresolved)
                    git_commit = capture(
                        `git -C $wt commit --no-edit -m "integration: $(b.name)"`;
                        nothing_on_error = true,
                    )
                    if git_commit !== nothing
                        println("  (resolved from rerere)")
                        continue
                    end
                end
                println("  CONFLICT")
                capture(`git -C $wt merge --abort`; nothing_on_error = true)
                conflict = b.name
                break
            end
            println()
        end
        if conflict !== nothing
            println("  stopped at `$conflict`; worktree left at $wt")
            push!(failed, "$(pkg.name):$(conflict)")
            continue
        end
        push && git(pkg, "push", "--force-with-lease", "origin", "integration")
        println("  integration = ", capture(`git -C $wt rev-parse --short HEAD`))
    end
    isempty(failed) || println("\nconflicted: ", join(failed, ", "))
    return failed
end

# ---------------------------------------------------------------------------------------------
# patch

"""
Regenerate `deps/patches/<package>.patch`: everything the vendored copy has that its
`integration` branch does not, once both sides have been pruned to what Ristretto actually ships. What
is left is the Ristretto-only adaptation -- relative imports, an inlined `ext/`, whatever else an
included submodule forces -- plus any fix that still owes itself to a branch, which is precisely
the drift that should be visible in one reviewable file instead of smeared through `deps/`.

Run it after `rebuild` and before the first `sync`, and again whenever a fix is carried back to
its branch: what the branch now contains leaves the patch on its own.
"""
function makepatch(packages)
    mkpath(joinpath(ROOT, "deps", "patches"))
    for pkg in packages
        require_checkout(pkg)
        wt = worktree(pkg)
        isdir(wt) || error("no integration worktree for $(pkg.name); run `rebuild` first")
        scratch = mktempdir()
        try
            # Both sides are laid out under the vendored prefix inside a scratch directory, so
            # that `diff --no-index` prints exactly the `<prefix>/file` paths `git apply` expects
            # from the repository root. A clean `git archive` export rather than the worktree
            # itself, because the worktree carries a `.git` file that the vendored copy does not
            # and that difference is not an adaptation.
            before = joinpath(scratch, "a", pkg.prefix)
            after = joinpath(scratch, "b", pkg.prefix)
            mkpath(before)
            mkpath(dirname(after))
            run(pipeline(`git -C $wt archive integration`, `tar -x -C $before`))
            prune!(before, pkg.prune)
            run(`cp -a $(joinpath(ROOT, pkg.prefix)) $after`)
            out = joinpath(ROOT, "deps", "patches", "$(pkg.name).patch")
            # Captured raw, not through `capture`: a patch's trailing bytes are significant.
            # `diff --no-index` exits non-zero exactly when the trees differ, which is the
            # expected case here, so the error path is the normal one.
            buf = IOBuffer()
            # `--no-renames`: with `--no-prefix` the scratch directory names `a`/`b` end up inside
            # the `rename from`/`rename to` lines, which `git apply` then cannot follow. A rename
            # written as a delete plus an add costs a few lines and always applies.
            cmd = `git -C $scratch diff --no-index --no-prefix --no-renames --binary a/$(pkg.prefix) b/$(pkg.prefix)`
            try
                run(pipeline(cmd; stdout = buf))
            catch
            end
            text = String(take!(buf))
            if isempty(strip(text))
                isfile(out) && rm(out)
                println(pkg.name, ": vendored copy matches integration exactly, no patch")
            else
                write(out, text)
                files = count(l -> startswith(l, "diff --git "), split(text, '\n'))
                println(pkg.name, ": ", files, " file(s) -> ", relpath(out, ROOT))
            end
        finally
            rm(scratch; recursive = true, force = true)
        end
    end
end

# ---------------------------------------------------------------------------------------------
# sync

"""
Project each package's `integration` branch into `deps/` with `git subtree pull --squash`, so the
vendored copy carries the revision it came from in its commit message and drift becomes
computable instead of guessed at. Ristretto-local adaptations that no upstream would take -- relative
imports, an `ext/` wired in by hand, because an included submodule loads neither -- live in
`deps/patches/<package>.patch` and are re-applied afterwards. A patch that stops applying has
been upstreamed and can be deleted.
"""
function sync(packages)
    for pkg in packages
        println("== ", pkg.name)
        # `deps/` is generated output: a hand edit here would be silently squashed away.
        dirty = here("status", "--porcelain", "--", pkg.prefix)
        isempty(dirty) || error(
            "uncommitted changes under $(pkg.prefix); commit or discard them -- `deps/` is " *
            "generated, fix bugs on the owning branch instead",
        )
        # `git subtree` is a contrib script some distributions leave out
        # (`git: 'subtree' is not a git command`); it must be on PATH for this to run --
        # install it from https://github.com/git/git/blob/<matching-tag>/contrib/subtree/git-subtree.sh
        # if it is missing. Using the real command, not a hand-rolled `read-tree --prefix`
        # projection, keeps the split-commit bookkeeping (`git subtree log`, `git subtree split`)
        # usable on this copy, not just the two trailers replayed from it. A package vendored for
        # the first time has no subtree to pull into yet, so it is added instead.
        first_time = !isdir(joinpath(ROOT, pkg.prefix))
        here(
            "subtree",
            first_time ? "add" : "pull",
            "--prefix",
            pkg.prefix,
            pkg.fork,
            "integration",
            "--squash",
            "-m",
            "chore($(pkg.name)): $(first_time ? "vendor" : "re-vendor") integration",
        )
        # The subtree pull brings the whole upstream tree; Ristretto keeps only what it compiles.
        if !isempty(pkg.prune)
            prune!(joinpath(ROOT, pkg.prefix), pkg.prune)
            here("add", "-A", "--", pkg.prefix)
            isempty(here("diff", "--cached", "--name-only", "--", pkg.prefix)) ||
                here("commit", "-m", "chore($(pkg.name)): prune what Ristretto does not ship")
        end
        patch = joinpath(ROOT, "deps", "patches", "$(pkg.name).patch")
        if isfile(patch)
            println("  apply ", relpath(patch, ROOT))
            here("apply", "--3way", patch)
        end
    end
end

# ---------------------------------------------------------------------------------------------
# status

json_escape(s) = replace(string(s), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n")
json(s::AbstractString) = "\"$(json_escape(s))\""
json(b::Bool) = b ? "true" : "false"
json(n::Integer) = string(n)
json(v::Vector) = "[" * join(json.(v), ",") * "]"
json(d::Vector{<:Pair}) = "{" * join(["$(json(k)):$(json(v))" for (k, v) in d], ",") * "}"

"""
The state of the whole stack, as the text summary a terminal wants or as the JSON the dashboard
reads. Regenerate the dashboard's data with

    julia deps/vendor.jl status --json > docs/src/assets/vendor-status.json
"""
function status(packages; as_json::Bool = false)
    entries = Vector{Pair{String,Any}}[]
    for pkg in packages
        for b in pkg.stack
            info = pr_state(b.pr)
            intended = isempty(b.parent) ? last(split(pkg.base, '/')) : b.parent
            state = if isempty(b.pr)
                "untracked"
            elseif info === nothing
                "unknown"
            elseif info.state == "MERGED"
                "merged"
            elseif info.state == "CLOSED"
                "closed"
            elseif info.base != intended
                "mis-based"
            elseif occursin("hakkelt/", b.pr)
                "fork"
            else
                "upstream"
            end
            push!(
                entries,
                [
                    "package" => pkg.name,
                    "branch" => b.name,
                    "parent" => b.parent,
                    "pr" => b.pr,
                    "url" => info === nothing ? "" : info.url,
                    "base_on_github" => info === nothing ? "" : info.base,
                    "intended_base" => String(intended),
                    "draft" => info === nothing ? false : info.draft,
                    "state" => state,
                ],
            )
        end
    end
    if as_json
        head = [
            "generated" => string(today()),
            "ristretto_commit" => here("rev-parse", "--short", "HEAD"),
            "gh" => gh_available(),
        ]
        println("{", join(["$(json(k)):$(json(v))" for (k, v) in head], ","), ",\"branches\":[")
        println(join(["  " * json(e) for e in entries], ",\n"))
        println("]}")
    else
        width = maximum(length(e[2][2]) for e in entries; init = 0)
        current = ""
        for e in entries
            pkg, branch, state = e[1][2], e[2][2], e[9][2]
            pkg == current || (println("== ", pkg); current = pkg)
            println("  ", rpad(branch, width), "  ", rpad(state, 10), "  ", e[4][2])
        end
        counts = Dict{String,Int}()
        for e in entries
            counts[e[9][2]] = get(counts, e[9][2], 0) + 1
        end
        println("\n", join(["$v $k" for (k, v) in sort(collect(counts))], " · "))
    end
end

# ---------------------------------------------------------------------------------------------

const USAGE = """
usage: julia deps/vendor.jl <command> [package...] [options]

  check [--no-fetch]               compare the manifest with GitHub and the local checkouts
  rebuild [--no-push] [--no-fetch] rebuild each fork's `integration` branch from the manifest
  patch                            regenerate deps/patches/<package>.patch from that branch
  sync                             project `integration` into deps/ via `git subtree pull --squash`
  status [--json]                  the state of the whole stack
"""

function main(args)
    isempty(args) && (print(USAGE); return 1)
    command = first(args)
    rest = collect(args[2:end])
    flags = filter(startswith("-"), rest)
    names = filter(!startswith("-"), rest)
    packages = select(load_manifest(), names)
    isempty(packages) && error("no package in $MANIFEST matches $(join(names, ", "))")
    if command == "check"
        return check(packages; fetch = !("--no-fetch" in flags)) == 0 ? 0 : 1
    elseif command == "rebuild"
        return isempty(
            rebuild(
                packages;
                push = !("--no-push" in flags),
                fetch = !("--no-fetch" in flags),
            ),
        ) ? 0 : 1
    elseif command == "patch"
        makepatch(packages)
    elseif command == "sync"
        sync(packages)
    elseif command == "status"
        status(packages; as_json = "--json" in flags)
    else
        print(USAGE)
        return 1
    end
    return 0
end

end # module

if abspath(PROGRAM_FILE) == (@__FILE__)
    exit(VendorTool.main(ARGS))
end
