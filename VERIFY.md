# Verifying this release

## 1 · Check the digest

```bash
shasum -a 256 paper/Assay_Whitepaper_EN.pdf     # macOS
sha256sum paper/Assay_Whitepaper_EN.pdf          # Linux
```

Expected:

```
38dcf534194081c2efdb53b56ac0a7a4bf0fa528137997c6a65f90449f5603b2
```

## 2 · Verify the Bitcoin timestamp

```bash
pip install opentimestamps-client
ots verify paper/Assay_Whitepaper_EN.pdf.ots
```

The proof in this repository is already **Bitcoin-attested** (block 963143). The
fastest check needs no installation at all: drag the PDF and the `.ots` file onto
https://opentimestamps.org. For a fully independent check, `ots verify` against your
own Bitcoin node. To refresh the proof with additional attestations:

```bash
ots upgrade paper/Assay_Whitepaper_EN.pdf.ots
```

Successful verification proves the PDF existed, byte-for-byte, at the attested time
— the paper's priority claim rests on this, not on trust in the author.
