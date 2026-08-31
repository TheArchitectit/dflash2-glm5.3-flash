# serving-security — shipped serving artifacts fail closed

## ADDED Requirements

### Requirement: REQ-SS1 Shipped units bind loopback by default

Systemd units shipped in `systemd/` SHALL bind the server to
`127.0.0.1` by default. Binding a non-loopback interface SHALL be an
explicit user edit of a commented opt-in line that documents the
acceptance in place: llama-server performs no authentication, so a
widely bound endpoint is usable by anyone who can reach it, and the
operator owns the firewalling.

#### Scenario: fresh unit

- **WHEN** a shipped unit is installed unmodified
- **THEN** llama-server listens on 127.0.0.1 only

#### Scenario: deliberate LAN exposure

- **WHEN** an operator uncomments the `0.0.0.0` opt-in line
- **THEN** the acceptance note (no auth, operator firewalls) is read at the moment of the edit, in the same file
