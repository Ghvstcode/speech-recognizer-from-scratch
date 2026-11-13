# Sample clips

A dozen short recordings so that every example in the book runs right after
`git clone`, before any dataset has been downloaded. They are unmodified
copies of files from two public datasets. The file names were changed; the
audio was not.

## From the Free Spoken Digit Dataset (8 kHz, 16-bit, mono)

| File | Original file | Speaker | Word |
|---|---|---|---|
| `zero_jackson.wav` | `recordings/0_jackson_0.wav` | jackson | "zero" |
| `zero_george.wav` | `recordings/0_george_0.wav` | george | "zero" |
| `three_jackson.wav` | `recordings/3_jackson_0.wav` | jackson | "three" |
| `three_george.wav` | `recordings/3_george_0.wav` | george | "three" |
| `seven_jackson.wav` | `recordings/7_jackson_0.wav` | jackson | "seven" |
| `seven_george.wav` | `recordings/7_george_0.wav` | george | "seven" |

Source: Zohar Jackson and contributors, *Free Spoken Digit Dataset (FSDD)*,
<https://github.com/Jakobovski/free-spoken-digit-dataset>.
License: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

These clips were trimmed by the dataset's authors so that there is almost no
silence before or after the word.

## From Google Speech Commands v0.02 (16 kHz, 16-bit, mono, one second each)

| File | Original file | Word |
|---|---|---|
| `yes.wav` | `yes/0137b3f4_nohash_0.wav` | "yes" |
| `no.wav` | `no/38d78313_nohash_3.wav` | "no" |
| `seven.wav` | `seven/0137b3f4_nohash_1.wav` | "seven" |
| `happy.wav` | `happy/3cc595de_nohash_0.wav` | "happy" |
| `stop.wav` | `stop/51055bda_nohash_2.wav` | "stop" |
| `go.wav` | `go/5f9cd2eb_nohash_0.wav` | "go" |

Source: Pete Warden / Google, *Speech Commands: A Dataset for
Limited-Vocabulary Speech Recognition*, 2018, <https://arxiv.org/abs/1804.03209>.
License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

These clips are one second long with the word somewhere in the middle and
room noise around it. They were picked from the dataset because they are not
clipped and have a low, but not artificially silent, background.

## Licensing of this directory

The files keep the licenses of the datasets they come from, as listed above.
The six FSDD clips are share-alike (CC BY-SA 4.0): if you redistribute them,
modified or not, you must do so under the same license. The rest of this
repository is licensed separately.
