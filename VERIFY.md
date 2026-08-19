# Verifying this release

## 1 · Check the digest

```bash
shasum -a 256 paper/Assay_Whitepaper_EN.pdf     # macOS
sha256sum paper/Assay_Whitepaper_EN.pdf          # Linux
```

Expected (v0.2.1, current):

```
3d102a75bf35c9982dcf459431f6e95c71d9ad8f0559ee0a93c3a116a7ab21ef
```

Archived v0.2 (`paper/archive/v0.2/Assay_Whitepaper_EN.pdf`):

```
38dcf534194081c2efdb53b56ac0a7a4bf0fa528137997c6a65f90449f5603b2
```

`paper/SHA256SUMS.txt` lists both; `shasum -c SHA256SUMS.txt` from inside `paper/`
checks everything at once.

## 2 · Verify the Bitcoin timestamp

```bash
pip install opentimestamps-client
ots verify paper/Assay_Whitepaper_EN.pdf.ots
ots verify paper/archive/v0.2/Assay_Whitepaper_EN.pdf.ots
```

The **v0.2 proof is Bitcoin-attested in block 963143** — that attestation is
permanent and establishes the priority date of the work. The **v0.2.1 proof** (an
editorial revision: the fig.-3 caption now separates trust-weighted votes from
verified identities) is freshly stamped; calendar aggregation and Bitcoin
confirmation typically complete within hours to a few days, and a scheduled
workflow in this repository upgrades the proof automatically. Until then,
`ots verify` on v0.2.1 reports the pending calendar attestations.

The fastest check needs no installation at all: drag a PDF and its `.ots` file onto
https://opentimestamps.org. For a fully independent check, run `ots verify` against
your own Bitcoin node. To refresh a proof by hand:

```bash
ots upgrade paper/Assay_Whitepaper_EN.pdf.ots
```

Successful verification proves the PDF existed, byte-for-byte, at the attested time
— the paper's priority claim rests on this, not on trust in the author.
