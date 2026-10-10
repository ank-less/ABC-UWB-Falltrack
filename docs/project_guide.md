# ABC 2027 UWB-FallTrack Project Guide

Updated 2026-10-08. This project is focused on the UWB-FallTrack challenge.

## Challenge focus

The stated task is continuous fall detection from UWB radar streams. A university announcement identifies the challenge organizer as Université du Québec à Chicoutimi (UQAC).

- ABC 2027 challenge page: <https://autocare.ai/abc2027/#challenge>
- UWB-FallTrack description and organizer reference: <https://informatika.unimal.ac.id/berita/abc-2027-challenges-peluang-kompetisi-computer-vision-dan-artificial-intelligence-tingkat-internasional>

## Confirm the official protocol before modeling

The official ABC page is the starting point, but the detailed UWB challenge protocol has not yet been verified here. Before implementing the submission pipeline, confirm from the challenge organizers:

- how to register and obtain the dataset;
- stream format, sampling rate, sensors/channels, and timestamp conventions;
- labels, event boundaries, and treatment of ambiguous or missing data;
- train/validation/test partitions and whether data may be segmented or resampled;
- required prediction format, evaluation metrics, and submission procedure;
- paper, code, and other deliverable requirements.

Do not assume a particular radar representation, label schema, split, or metric until the official materials specify it.

## Initial technical workflow

1. Obtain the official data and usage terms.
2. Inventory files, schemas, sampling rates, labels, missingness, and class balance.
3. Build a reproducible preprocessing and event-window pipeline; preserve chronological order and split by subject/session as required by the rules.
4. Establish a simple signal-feature baseline, then compare temporal models appropriate to the released representation.
5. Evaluate with the official metrics and inspect missed falls, false alarms, and detection delay where the protocol supports those analyses.
6. Generate and validate a submission in the exact required format.
