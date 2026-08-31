# publish-safety — outward actions are deliberate

## ADDED Requirements

### Requirement: REQ-PS1 Visibility flips require explicit confirmation

Any script step that makes a published artifact public (HF repo
visibility, equivalent outward flips) SHALL require an explicit typed
confirmation before executing, defaulting to deny on absent, empty, or
non-matching input (including non-interactive EOF). A non-interactive
escape (e.g. `FLIP_PUBLIC=1`) MAY exist for scripted runs. Aborting
SHALL leave the artifact private and exit nonzero, naming the escape
variable.

#### Scenario: unconfirmed flip aborts

- **WHEN** `upload.sh` reaches step [4/4] with no input (EOF) or any answer other than the exact confirmation word
- **THEN** the script prints an abort message naming `FLIP_PUBLIC=1`, exits 1, and the repo remains private

#### Scenario: scripted flip is explicit

- **WHEN** the script is re-run with `FLIP_PUBLIC=1` after its gates passed
- **THEN** the visibility flip proceeds without prompting
