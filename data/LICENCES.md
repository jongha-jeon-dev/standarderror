# What is in here, and why some of it is not

This directory holds source data downloaded by hand, because the container that
runs these experiments cannot reach data APIs. Whether a file may be committed to
a public repository is a licensing question with a different answer per source, so
the answer is written down here rather than remembered.

The rule is narrow on purpose: **publish a file only when its provider says in
writing that redistribution is allowed.** Everything else stays out of git, and
the post that uses it publishes statistics computed from it rather than values —
see `docs/HOUSE_STYLE.md`. A blanket exclusion would be simpler and would also
make several posts impossible to reproduce, which is the cost being paid here.

## Committed

### `worldbank/` — World Bank Open Data, CC BY 4.0

The World Bank's licence for its Open Data catalogue "allows users to copy, modify
and distribute data in any format for any purpose, including commercial use",
requiring that users "give appropriate credit (attribution) and indicate if they
have made any changes, including translations."
<https://datacatalog.worldbank.org/public-licenses>

The files here are the original download archives, **unmodified**, including the
provider's own `Metadata_Indicator_*` and `Metadata_Country_*` sheets. Any
transformation happens in code at read time and is visible in the experiment that
does it.

Attribution: World Bank, World Development Indicators. Licensed under CC BY 4.0.

*Not* covered by that licence, and therefore never committed here: World Bank
microdata (unit-level survey and census records), which carry the Microdata
Research License and forbid redistribution without written consent, and any
dataset the catalogue labels ODbL or custom.

### `us_labor/` — US Bureau of Labor Statistics, public domain

BLS states that "everything that we publish, both in hard copy and electronically,
is in the public domain" and that "you are free to use our public domain material
without specific permission, although we do ask that you cite the Bureau of Labor
Statistics as the source." <https://www.bls.gov/bls/linksite.htm>

Works of the US federal government carry no domestic copyright (17 U.S.C. 105).
The exception BLS names is previously copyrighted photographs and illustrations,
which does not reach a data table.

Attribution: US Bureau of Labor Statistics, Current Population Survey.

## Not committed, and why

### `fred/` — index values under the provider's terms

NASDAQ Composite, Nikkei 225 and VIX levels reach this repository through FRED,
but the underlying index values belong to their providers and FRED redistributes
them under terms that do not extend to onward publication. `fred/README.md` holds
the download URLs so anyone can fetch the identical files, and `fred/VINTAGE.txt`
holds row counts and SHA-256 digests so they can confirm they got the same ones.
Posts publish statistics, never values.

### `korea/` — mixed, and it also contains copies of the FRED files

KOSIS and ECOS exports, plus — importantly — duplicates of the FRED series above.
Anyone tempted to relax the rule for this directory should notice that second part
first.

### `korea_power/` — 공공누리 (KOGL), typed per table

Korea Power Exchange EPSIS exports. Korean public data is licensed under KOGL, but
the *type* is set per table and types 2 to 4 restrict commercial use or
modification. Until the type of each specific table is confirmed, the tables stay
out and the posts publish computed statistics — which is what the published
exp021 already says it does.

## Provenance

Downloaded 2026-08-25 and 2026-08-26. Digests so a reader can confirm they have
the same bytes:

| file | bytes | sha256 |
|---|---|---|
| `data/worldbank/API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_102971.zip` | 39,840 | `8c9b3e337af49f10bb2f40c2090c6000485108aac2e39a5bdfd584021e17d46f` |
| `data/worldbank/API_EN.GHG.CO2.PC.CE.AR5_DS2_en_csv_v2_34040.zip` | 133,699 | `db1ad974d61ca1cb27ad9888167925a1e2a153ff303f7ed141a1c8ae320f0d53` |
| `data/worldbank/API_FS.AST.PRVT.GD.ZS_DS2_en_csv_v2_34735.zip` | 103,246 | `64f0bfe7bc7189e87f16379653c28bdc964ff2cd4027ffa7e2ad8f7c4902e467` |
| `data/worldbank/API_GB.XPD.RSDV.GD.ZS_DS2_en_csv_v2_232.zip` | 30,940 | `e87e687476483139c6b0613e023d64438e72bb04e0c0cefad09074db03774d2c` |
| `data/worldbank/API_NY.GDP.PCAP.PP.KD_DS2_en_csv_v2_33608.zip` | 93,802 | `7ef5882fb5a4417977d2de1236fb99e47e3d6982c1ef2c6eb57550d56e665c87` |
| `data/worldbank/API_SP.DYN.LE00.IN_DS2_en_csv_v2_408.zip` | 90,441 | `80890a2f860714e5fd9c19368ce328c6cadf2a66a5d70b104999f3f1f22fed7d` |
| `data/worldbank/API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_33381.zip` | 73,845 | `0eb0d9430eec710154f7193f9dd765fc174169c58b50cc397d955e23ca3857f9` |
| `data/worldbank/API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_33901.zip` | 159,234 | `b518ecc9b8eb37437183acc2316d9e55a45146c411a39b2741cfdd046e8043e2` |
| `data/us_labor/LNS14000000.xlsx` | 10,531 | `a2b17fead3cbe080d8b3e8129aaff61e1a3119e3dccccbbe9a63bef58a992047` |
| `data/us_labor/LNU04000000.xlsx` | 10,818 | `3d83a4d319c86a3e1bf1ec473dfe4e635a24bb410df0e8b65e36f51755df7e12` |
