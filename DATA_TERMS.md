# Data Provenance and Use

This is a reproducibility release of derived experimental inputs. The source charging data were obtained from [ACN-Data](https://ev.caltech.edu/dataset). Raw API exports, ACN access tokens, user identifiers, station identifiers, and original session identifier strings are not included.

The public snapshot replaces session identifiers and calendar dates while preserving the numerical inputs used by every detector. Absolute dates are synthetic and must not be used for claims about calendar-time trends, seasons, or external temporal validation. Time-of-day aggregate power calculations are preserved. Pseudonymization is not a guarantee of irreversible anonymity.

The original dataset remains subject to its provider's terms and attribution requirements. Public access to a dataset is not a blanket redistribution license. This repository does not claim ownership of, or grant a new license to, underlying ACN data. Obtain original records and current terms directly from the official portal when needed; do not commit your token or raw export here.

Research code and results are supplied for inspection and reproducibility. No blanket license is assigned to code, reference material, or data in this initial release. A public repository is not by itself a grant of unrestricted reuse rights; obtain permission from the relevant rights holders where necessary. The authors can add an appropriate explicit software license later.

All attacks in this benchmark are simulations. Use the artifacts for defensive research; the release contains no infrastructure access credentials or live charging-system targets.
