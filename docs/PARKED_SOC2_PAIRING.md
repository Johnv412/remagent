# PARKED — SOC2 scanner × RemAgent pairing

**Status:** parked · **Dated:** 2026-10-15 · **Do not build until after the 1.1.2 launch.**

## The idea

Store SOC2 scanner findings as brain facts, and treat remediation as supersession.

- A finding becomes a fact: `entity` = the audited GCP project or control,
  `attribute` = the control id, `value` = the finding state.
- Fixing it is not a delete. A new active fact supersedes the old one, and
  the old one stays inactive with `superseded_by` pointing at the fix — the
  same chain RemAgent already keeps for every other fact.
- The scanner gains memory of past audits; the brain gains a live view of
  which findings are still open, which is what the P1 rule "promptly
  remediate and close all SOC 2 findings" needs in order to be checkable.

## Why it is parked

- Nothing about it is required for launch, and it touches two systems.
- The brain already holds 211 facts about the scanner and one closed finding
  (`SOC 2 finding (First).status = remediated and closed`), so the shape is
  proven in miniature. There is no urgency to generalise it now.
- The scanner is a stronger open-source candidate on its own; coupling the two
  before either has shipped would blur both.

## Not decided

Which direction writes first (scanner → brain, or brain → scanner), and whether
findings are logged as raw turns for the dream to consolidate or written as
facts through a dedicated path. Decide after launch, with real audit docs.
