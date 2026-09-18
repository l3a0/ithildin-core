# Design — quant-core

## Contents

- [Premise](#premise)
- [Vocabulary](#vocabulary)
- [Configuration](#configuration)
- [Considered and rejected](#considered-and-rejected)

## Premise

**A change to a function body here is irreversible for the consumer.**

Consuming repositories commit numbers computed from this code. They go into
test assertions, into committed ledgers, and into published documents that
other people read. This package can be re-read at any commit, so nothing about
it is lost. A number already published from it cannot be un-published.

That asymmetry is the whole design. Everything about this package is
regenerable except the effect it has already had somewhere else, so the
expensive operation is not losing code, it is moving a number that has left
the building.

Two consequences follow, and they pull in opposite directions on purpose.

1. **Adding a function is cheap.** It moves nothing, so it needs no
   coordination and no consumer sign-off.
2. **Editing a function body is expensive.** It is a re-pin of every consumer,
   and the version number is the mechanism that lets each one absorb it when
   it is ready rather than when this repo happens to push.

A consumer that floats its dependency gives up the only protection this design
offers. So the install instructions pin a tag, and both consuming repositories
commit the resolved commit.

### Why a shared package rather than two copies kept in step

The alternative was a rule: keep `common/timeseries.py` and
`src/chan/timeseries.py` matching, with a test to catch divergence. It was
rejected because the test cannot exist. Neither repository's continuous
integration can see the other's checkout, so the check would have to compare
against a committed checksum, which fires only when somebody remembers to
update it. A rule that depends on remembering is the thing the duplication
already was.

## Vocabulary

Terms with exact definitions, reused on purpose. A term listed here is not a
candidate for a synonym. Anything not listed here follows the writing-style
rule in [CLAUDE.md](../CLAUDE.md), which prefers deleting an in-group term to
glossing it.

| Term | Definition |
| --- | --- |
| Consumer | A repository that installs this package. There are two: [trading-strategies](https://github.com/l3a0/trading-strategies) and [quantitative-trading](https://github.com/l3a0/quantitative-trading). A call site inside this repo is not a consumer. |
| Re-pin | Updating a committed number because the code that produced it changed. The work lands in the consumer, not here. |
| Body change | An edit to what an existing function computes, including a change that is arithmetically equivalent on paper. Reordering a sum is a body change, because floating-point addition is not associative. |
| Load-bearing default | An argument default that decides an answer rather than saving typing. `adf_tstat`'s fixed lag and `ols`'s absent intercept are the two here. Changing one is a body change for every caller that did not pass the argument. |

## Configuration

This repository is public. Tracked files never carry secrets or
machine-specific paths.

There is no configuration, and that is a property worth protecting rather than
a stage this package has not reached. Every function takes arrays and returns
numbers. Nothing reads a file, an environment variable, or a network. An
estimator that read a setting could return two different answers to the same
call, and a consumer pinning its result would have pinned the setting without
knowing it.

So there is no configuration table. A future module that needs one needs this
section rewritten first, and the rejection below argues against it.

## Considered and rejected

Machinery that was considered and cut, pinned here with the reason. The point
is that a decision stays decided. When something new is cut, add it here in
the same change that cuts it.

Reviews armor what exists and rarely ask whether it should exist, so this
register is also where the answer to "why is this needed" gets written down
once rather than re-argued.

| Cut | Why |
| --- | --- |
| Keeping two copies in step with a drift test | The test cannot exist. Neither consumer's continuous integration can see the other's checkout, so the check would compare against a committed checksum that fires only when somebody updates it. A rule that depends on remembering is what the duplication already was. |
| Reading configuration or files from an estimator | It would let one call return two answers, and a consumer pinning the result would have pinned the setting without knowing it. Every function takes arrays and returns numbers. A caller that needs a setting resolves it and passes the value. |
| Moving `paths.py` here | Its 47 consumers next door look like the strongest case in the account, but the line that matters differs between the two repos: `parents[1]` there against `parents[2]` here, because one nests its package under `src/`. What is shared is the pattern, not the constant, and a helper thin enough to be shared is thinner than the import that would reach it. |
| Moving `position_sizing.py` and `trade_ledger.py` here | Each has consumers in one repository and none in the other, which is the bar in [CLAUDE.md](../CLAUDE.md) failing. Both also carry that repo's own vocabulary for per-trade outcomes, so they would need rewriting to be neutral before they could be shared, and nothing would consume the result. |
| A single `quantcore.all` convenience import | It would make every consumer's import graph depend on every module here, so a new dependency in one module becomes a new dependency everywhere. Import the module you use. |
