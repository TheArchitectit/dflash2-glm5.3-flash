# benchmark — CPU performance measurement and publication

## ADDED Requirements

### Requirement: REQ-BM-1 Solo CPU benchmark

The benchmark SHALL run on ucs03 solo — never concurrent with the production `:8086` service — and SHALL report: tokens/sec vs the 1.32 t/s baseline, acceptance %, and wall-clock time on the standard 3-task agentic suite.

#### Scenario: benchmark run solo

- **WHEN** the benchmark is executed
- **THEN** it runs solo on ucs03 with the production `:8086` service idle, producing t/s, acceptance %, and wall-clock numbers for the 3-task agentic suite

#### Scenario: t/s improvement reported

- **WHEN** the benchmark completes
- **THEN** effective tokens/sec is reported against the 1.32 t/s baseline (projection: ~7 t/s at the published 5.78 accepted tokens/step)

### Requirement: REQ-BM-2 Publication

The converted GGUF SHALL be published to Hugging Face with CC BY-NC-ND 4.0 attribution (non-commercial; the DFlash2 weights' license). Release notes SHALL be provided to incoai and to llama.cpp.

#### Scenario: upload with license attribution

- **WHEN** the GGUF is uploaded to HF
- **THEN** the upload carries CC BY-NC-ND 4.0 attribution and notes are sent to incoai and llama.cpp
