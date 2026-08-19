# Verifying this release

## 1 · Check the digest

```bash
shasum -a 256 paper/Assay_Whitepaper_EN.pdf     # macOS
sha256sum paper/Assay_Whitepaper_EN.pdf          # Linux
```

Expected (v0.2.2, current):

```
78cb8dc1720fddde060e8bf531c0d5e1a1e1e0a053e468ff74a23575c930285b
```

Superseded versions are archived with their proofs:

```
3d102a75bf35c9982dcf459431f6e95c71d9ad8f0559ee0a93c3a116a7ab21ef  archive/v0.2.1/
38dcf534194081c2efdb53b56ac0a7a4bf0fa528137997c6a65f90449f5603b2  archive/v0.2/
```

`paper/SHA256SUMS.txt` lists all three; `shasum -c SHA256SUMS.txt` from inside
`paper/` checks everything at once.

## 2 · Verify the Bitcoin timestamp

```bash
pip install opentimestamps-client
ots verify paper/Assay_Whitepaper_EN.pdf.ots
ots verify paper/archive/v0.2/Assay_Whitepaper_EN.pdf.ots
```

The **v0.2 proof is Bitcoin-attested in block 963143** — that attestation is
permanent and establishes the priority date of the work. Newer proofs are
stamped at release and upgraded automatically by a scheduled workflow in this
repository; calendar aggregation and Bitcoin confirmation typically complete
within hours to a few days. Until then, `ots verify` on a fresh proof reports
the pending calendar attestations.

The fastest check needs no installation at all: drag a PDF and its `.ots` file onto
https://opentimestamps.org. For a fully independent check, run `ots verify` against
your own Bitcoin node. To refresh a proof by hand:

```bash
ots upgrade paper/Assay_Whitepaper_EN.pdf.ots
```

## Revision history

- **v0.2.2** — corrects the heavy-contamination detection figure from a rounded
  "100%" to the measured **99.7% (598 of 600 replications)**; the robustness
  section now reports the accuracy-scored comparison it tests and states the
  IRT-EAP ablation explicitly; reproduction paths in `poc/` fixed so a fresh
  clone runs end-to-end (the arena corpus now downloads itself).
- **v0.2.1** — fig. 3 caption separates trust-weighted votes (~687) from
  distinct verified identities (~138, ≈ €414); manipulation cost stated as a
  floor at PoC scale; footer changed to public working paper.
- **v0.2** — first published version; Bitcoin-attested in block 963143.

Successful verification proves the PDF existed, byte-for-byte, at the attested time
— the paper's priority claim rests on this, not on trust in the author.
