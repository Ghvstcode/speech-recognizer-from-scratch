//! Scoring a recognizer: edit distance, word error rate, character error rate.
//!
//! A recognizer's output is rarely exactly right or exactly wrong. "turn the
//! lights of" against "turn the lights off" is one small mistake, and we want
//! a number that says so. The standard number is the **edit distance** (also
//! called Levenshtein distance): the smallest count of single-token edits that
//! turns one sequence into the other, where an edit is one of
//!
//! * a **substitution**: one token replaced by another,
//! * a **deletion**: a reference token the recognizer missed,
//! * an **insertion**: a token the recognizer made up.
//!
//! Divide the edit distance by the length of the reference and you get an
//! error rate. With words as tokens it is the word error rate (WER); with
//! characters as tokens it is the character error rate (CER).
//!
//! If you have read the dynamic time warping chapter this will feel familiar:
//! it is the same grid, the same three neighbours and the same "cheapest way
//! to get here" recurrence. DTW aligns two sequences of feature frames and
//! pays a distance per cell; edit distance aligns two sequences of symbols
//! and pays 0 or 1 per step.

/// The smallest number of substitutions, insertions and deletions that turns
/// `a` into `b`.
///
/// The recurrence, with `d[i][j]` meaning "the distance between the first `i`
/// tokens of `a` and the first `j` tokens of `b`":
///
/// ```text
/// d[0][j] = j                        (insert j tokens into nothing)
/// d[i][0] = i                        (delete i tokens)
/// d[i][j] = min( d[i-1][j-1] + cost, (match if equal, else substitute)
///                d[i-1][j]   + 1,    (delete a[i-1])
///                d[i][j-1]   + 1 )   (insert b[j-1])
///           where cost = 0 if a[i-1] == b[j-1], else 1
/// ```
///
/// Each row only looks at the row above it, so we keep two rows instead of
/// the whole table.
pub fn edit_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    // Row 0 of the table: d[0][j] = j.
    let mut previous: Vec<usize> = (0..=b.len()).collect();
    let mut current = vec![0; b.len() + 1];
    for i in 1..=a.len() {
        current[0] = i; // d[i][0] = i
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let diagonal = previous[j - 1] + cost;
            let delete = previous[j] + 1;
            let insert = current[j - 1] + 1;
            current[j] = diagonal.min(delete).min(insert);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[b.len()]
}

/// One step of an alignment between a reference and a hypothesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditOp {
    /// The reference token and the hypothesis token are equal.
    Match,
    /// The hypothesis has a different token where the reference has one.
    Substitute,
    /// The hypothesis has an extra token the reference does not have.
    Insert,
    /// The hypothesis is missing a token the reference has.
    Delete,
}

/// The cheapest alignment of `hypothesis` against `reference`, as the list of
/// operations read left to right.
///
/// `Match` and `Substitute` consume one token from each side, `Delete`
/// consumes one reference token, `Insert` consumes one hypothesis token. The
/// number of operations that are not `Match` equals
/// `edit_distance(reference, hypothesis)`.
///
/// This time we need the whole table, because after filling it we walk back
/// from the bottom-right corner to the top-left one asking at every cell
/// "which neighbour did my value come from?".
///
/// There is usually more than one cheapest alignment, and some read better
/// than others. "turn the lights off" against "turn light off now" is three
/// edits either as *delete "the", lights -> light, insert "now"* or as three
/// substitutions in a row (the -> light, lights -> off, off -> now). The
/// first keeps "off" lined up with "off" and is the one a person would
/// write. To get it we make a substitution cost a hair more than an insertion
/// or a deletion: with `unit = min(n, m) + 1`, insertions and deletions cost
/// `unit` and substitutions cost `unit + 1`. An alignment has at most
/// `min(n, m)` substitutions, so the extra `+ 1`s can never add up to a whole
/// `unit`: the total is still smallest for the alignments with the fewest
/// edits, and among those it is smallest for the one with the fewest
/// substitutions (which is the one with the most matches).
pub fn align<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> Vec<EditOp> {
    let (n, m) = (reference.len(), hypothesis.len());
    let unit = n.min(m) + 1;
    let step_cost = |i: usize, j: usize| if reference[i - 1] == hypothesis[j - 1] { 0 } else { unit + 1 };
    let width = m + 1;
    // d[i * width + j] = weighted cost of aligning reference[..i] with hypothesis[..j].
    let mut d = vec![0usize; (n + 1) * width];
    // Row 0: turning nothing into j tokens takes j insertions.
    for (j, cell) in d[..=m].iter_mut().enumerate() {
        *cell = j * unit;
    }
    for i in 1..=n {
        d[i * width] = i * unit;
        for j in 1..=m {
            let diagonal = d[(i - 1) * width + j - 1] + step_cost(i, j);
            let delete = d[(i - 1) * width + j] + unit;
            let insert = d[i * width + j - 1] + unit;
            d[i * width + j] = diagonal.min(delete).min(insert);
        }
    }

    // Walk back from (n, m) to (0, 0), collecting operations in reverse.
    let mut ops = Vec::with_capacity(n.max(m));
    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        let here = d[i * width + j];
        if i > 0 && j > 0 && here == d[(i - 1) * width + j - 1] + step_cost(i, j) {
            let same = reference[i - 1] == hypothesis[j - 1];
            ops.push(if same { EditOp::Match } else { EditOp::Substitute });
            i -= 1;
            j -= 1;
        } else if i > 0 && here == d[(i - 1) * width + j] + unit {
            ops.push(EditOp::Delete);
            i -= 1;
        } else {
            ops.push(EditOp::Insert);
            j -= 1;
        }
    }
    ops.reverse();
    ops
}

/// How many of each operation an alignment contains.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EditCounts {
    /// Tokens that were right.
    pub matches: usize,
    /// Tokens replaced by a different token.
    pub substitutions: usize,
    /// Extra tokens in the hypothesis.
    pub insertions: usize,
    /// Reference tokens missing from the hypothesis.
    pub deletions: usize,
}

impl EditCounts {
    /// Count the operations in an alignment produced by [`align`].
    pub fn from_ops(ops: &[EditOp]) -> Self {
        let mut counts = EditCounts::default();
        for op in ops {
            match op {
                EditOp::Match => counts.matches += 1,
                EditOp::Substitute => counts.substitutions += 1,
                EditOp::Insert => counts.insertions += 1,
                EditOp::Delete => counts.deletions += 1,
            }
        }
        counts
    }

    /// Substitutions + insertions + deletions: the edit distance.
    pub fn edits(&self) -> usize {
        self.substitutions + self.insertions + self.deletions
    }
}

/// Split a sentence into words: lower-cased, separated by any whitespace.
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(|w| w.to_lowercase()).collect()
}

/// Split a sentence into characters for CER: lower-cased, with every run of
/// whitespace squeezed to one space and none at the ends. Spaces count as
/// characters, so gluing two words together ("turnon") costs one edit.
pub fn chars(text: &str) -> Vec<char> {
    words(text).join(" ").chars().collect()
}

/// An error rate: `edits / reference_len`.
///
/// An empty reference has no sensible rate, so we divide by 1 instead of 0:
/// an empty hypothesis scores 0.0 and every invented token costs 1.0.
fn rate(edits: usize, reference_len: usize) -> f32 {
    edits as f32 / reference_len.max(1) as f32
}

/// Word error rate of one hypothesis against its reference:
/// (substitutions + insertions + deletions) / (words in the reference).
///
/// 0.0 is perfect. It is **not** capped at 1.0: a hypothesis with many extra
/// words can make more errors than the reference has words.
pub fn wer(reference: &str, hypothesis: &str) -> f32 {
    let (r, h) = (words(reference), words(hypothesis));
    rate(edit_distance(&r, &h), r.len())
}

/// Character error rate of one hypothesis against its reference: the same
/// formula as [`wer`], with characters (see [`chars`]) as the tokens.
pub fn cer(reference: &str, hypothesis: &str) -> f32 {
    let (r, h) = (chars(reference), chars(hypothesis));
    rate(edit_distance(&r, &h), r.len())
}

/// Running totals for scoring a whole test set.
///
/// The corpus-level rate is (all edits) / (all reference tokens). That is
/// *not* the average of the per-utterance rates, and the difference matters.
/// Take two utterances: "yes" recognized as "no" (1 error in 1 word, 100%)
/// and a ten-word sentence recognized perfectly (0%). The mean of the two
/// rates says 50%; but the recognizer got 1 word wrong out of 11, which is
/// 9%. Averaging rates lets short utterances count as much as long ones;
/// summing edits weights every word equally, which is what "word error
/// rate" promises.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ErrorTally {
    /// Utterances added so far.
    pub utterances: usize,
    /// Utterances whose hypothesis matched the reference word for word.
    pub exact_matches: usize,
    /// Sum of word-level edit distances.
    pub word_edits: usize,
    /// Sum of reference lengths in words.
    pub reference_words: usize,
    /// Sum of character-level edit distances.
    pub char_edits: usize,
    /// Sum of reference lengths in characters.
    pub reference_chars: usize,
}

impl ErrorTally {
    /// An empty tally.
    pub fn new() -> Self {
        Self::default()
    }

    /// Score one utterance and add it to the totals.
    pub fn add(&mut self, reference: &str, hypothesis: &str) {
        let (ref_words, hyp_words) = (words(reference), words(hypothesis));
        let (ref_chars, hyp_chars) = (chars(reference), chars(hypothesis));
        let word_edits = edit_distance(&ref_words, &hyp_words);
        self.utterances += 1;
        self.exact_matches += usize::from(word_edits == 0);
        self.word_edits += word_edits;
        self.reference_words += ref_words.len();
        self.char_edits += edit_distance(&ref_chars, &hyp_chars);
        self.reference_chars += ref_chars.len();
    }

    /// Fold another tally into this one (for example, one per worker thread).
    pub fn merge(&mut self, other: &ErrorTally) {
        self.utterances += other.utterances;
        self.exact_matches += other.exact_matches;
        self.word_edits += other.word_edits;
        self.reference_words += other.reference_words;
        self.char_edits += other.char_edits;
        self.reference_chars += other.reference_chars;
    }

    /// Corpus word error rate: total word edits / total reference words.
    pub fn wer(&self) -> f32 {
        rate(self.word_edits, self.reference_words)
    }

    /// Corpus character error rate: total character edits / total reference characters.
    pub fn cer(&self) -> f32 {
        rate(self.char_edits, self.reference_chars)
    }

    /// Fraction of utterances that were recognized with no word errors at all.
    pub fn exact_match_rate(&self) -> f32 {
        self.exact_matches as f32 / self.utterances.max(1) as f32
    }
}

/// Print a reference and a hypothesis one above the other, word by word, with
/// a third line marking what went wrong: `S` substitution, `I` insertion,
/// `D` deletion. Missing words are shown as a run of `*`.
///
/// ```text
/// REF: turn the lights off ***
/// HYP: turn *** light  off now
///           D   S          I
/// ```
pub fn format_alignment(reference: &str, hypothesis: &str) -> String {
    let (ref_words, hyp_words) = (words(reference), words(hypothesis));
    let ops = align(&ref_words, &hyp_words);
    let (mut ref_line, mut hyp_line, mut op_line) = (String::from("REF:"), String::from("HYP:"), String::from("    "));
    let (mut i, mut j) = (0, 0);
    for op in ops {
        // Which word (if any) each side shows in this column, and the marker under it.
        let (ref_word, hyp_word, mark) = match op {
            EditOp::Match => (Some(&ref_words[i]), Some(&hyp_words[j]), ""),
            EditOp::Substitute => (Some(&ref_words[i]), Some(&hyp_words[j]), "S"),
            EditOp::Delete => (Some(&ref_words[i]), None, "D"),
            EditOp::Insert => (None, Some(&hyp_words[j]), "I"),
        };
        i += usize::from(ref_word.is_some());
        j += usize::from(hyp_word.is_some());
        let width = ref_word.map_or(0, |w| w.chars().count()).max(hyp_word.map_or(0, |w| w.chars().count()));
        let stars = "*".repeat(width);
        ref_line.push_str(&format!(" {:<width$}", ref_word.unwrap_or(&stars)));
        hyp_line.push_str(&format!(" {:<width$}", hyp_word.unwrap_or(&stars)));
        op_line.push_str(&format!(" {:<width$}", mark));
    }
    format!("{}\n{}\n{}", ref_line.trim_end(), hyp_line.trim_end(), op_line.trim_end())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn letters(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn textbook_distances() {
        // kitten -> sitten (s/k) -> sittin (i/e) -> sitting (+g): 3 edits.
        assert_eq!(edit_distance(&letters("kitten"), &letters("sitting")), 3);
        // The other classic: saturday / sunday is 3 (delete a, delete t, r -> n).
        assert_eq!(edit_distance(&letters("saturday"), &letters("sunday")), 3);
        assert_eq!(edit_distance(&letters("flaw"), &letters("lawn")), 2);
        assert_eq!(edit_distance(&letters("same"), &letters("same")), 0);
    }

    #[test]
    fn empty_sequences() {
        let empty: Vec<char> = Vec::new();
        assert_eq!(edit_distance(&empty, &empty), 0);
        assert_eq!(edit_distance(&empty, &letters("abc")), 3);
        assert_eq!(edit_distance(&letters("abc"), &empty), 3);
        assert_eq!(align(&empty, &letters("ab")), vec![EditOp::Insert, EditOp::Insert]);
        assert_eq!(align(&letters("ab"), &empty), vec![EditOp::Delete, EditOp::Delete]);
        assert!(align(&empty, &empty).is_empty());
    }

    #[test]
    fn distance_is_symmetric() {
        let (a, b) = (letters("intention"), letters("execution"));
        assert_eq!(edit_distance(&a, &b), 5);
        assert_eq!(edit_distance(&b, &a), 5);
    }

    #[test]
    fn alignment_spells_out_kitten_sitting() {
        use EditOp::*;
        let ops = align(&letters("kitten"), &letters("sitting"));
        // k i t t e n -
        // s i t t i n g
        assert_eq!(ops, vec![Substitute, Match, Match, Match, Substitute, Match, Insert]);
    }

    /// Replaying an alignment must consume both sequences exactly, its
    /// matches must really be equal tokens, and its edit count must equal the
    /// edit distance.
    #[test]
    fn alignments_are_consistent_with_the_distance() {
        let pairs = [("kitten", "sitting"), ("saturday", "sunday"), ("abc", ""), ("", "xyz"), ("abcdef", "azced"), ("aaaa", "aa")];
        for (r, h) in pairs {
            let (r, h) = (letters(r), letters(h));
            let ops = align(&r, &h);
            let (mut i, mut j) = (0, 0);
            for op in &ops {
                match op {
                    EditOp::Match => {
                        assert!(r[i] == h[j]);
                        i += 1;
                        j += 1;
                    }
                    EditOp::Substitute => {
                        assert!(r[i] != h[j]);
                        i += 1;
                        j += 1;
                    }
                    EditOp::Delete => i += 1,
                    EditOp::Insert => j += 1,
                }
            }
            assert_eq!((i, j), (r.len(), h.len()));
            assert_eq!(EditCounts::from_ops(&ops).edits(), edit_distance(&r, &h));
        }
    }

    #[test]
    fn wer_counts_word_errors() {
        // One substitution (lights -> light) and one deletion (the): 2 / 4.
        assert_eq!(wer("turn the lights off", "turn light off"), 0.5);
        assert_eq!(wer("go left", "go left"), 0.0);
        // Case and extra whitespace do not count as errors.
        assert_eq!(wer("Go  Left", " go left "), 0.0);
    }

    #[test]
    fn wer_can_exceed_one() {
        // Reference has one word; the hypothesis substitutes it and adds two.
        assert_eq!(wer("yes", "no no no"), 3.0);
    }

    #[test]
    fn rates_with_empty_strings() {
        assert_eq!(wer("", ""), 0.0);
        assert_eq!(cer("", ""), 0.0);
        assert_eq!(wer("stop", ""), 1.0); // everything deleted
        assert_eq!(wer("", "stop now"), 2.0); // divided by 1, not by 0
    }

    #[test]
    fn cer_counts_character_errors() {
        // "sitting" vs "kitten": 3 edits over 7 reference characters.
        assert!((cer("sitting", "kitten") - 3.0 / 7.0).abs() < 1e-6);
        // A missing space is one character edit out of 7 ("go left").
        assert!((cer("go left", "goleft") - 1.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn corpus_rate_is_not_the_mean_of_rates() {
        let mut tally = ErrorTally::new();
        tally.add("yes", "no");
        tally.add("one two three four five six seven eight nine zero", "one two three four five six seven eight nine zero");
        // 1 wrong word out of 11, not (100% + 0%) / 2.
        assert!((tally.wer() - 1.0 / 11.0).abs() < 1e-6);
        assert_eq!(tally.exact_match_rate(), 0.5);
        // "yes" -> "no" is 3 character edits (y->n, e->o, delete s); the long
        // reference has 49 characters including its 9 spaces.
        assert_eq!((tally.char_edits, tally.reference_chars), (3, 3 + 49));

        let mut merged = ErrorTally::new();
        merged.merge(&tally);
        merged.merge(&tally);
        assert_eq!(merged.wer(), tally.wer());
        assert_eq!(merged.utterances, 4);
    }

    #[test]
    fn formatted_alignment_lines_up() {
        let text = format_alignment("turn the lights off", "turn light off now");
        let expected = "REF: turn the lights off ***\nHYP: turn *** light  off now\n          D   S          I";
        assert_eq!(text, expected);
    }
}
