//! Decoding: from a grid of per-frame probabilities to a sequence of labels.
//!
//! A CTC-trained network gives us, for each of T frames, a probability for
//! each of C classes (class 0 is the blank). Training summed over every
//! alignment of a *known* target. Decoding is the reverse problem: no target
//! is given, and we want the labeling that is most probable.
//!
//! Three decoders live here, from simple to careful:
//!
//! * [`greedy`]: take the most probable class in every frame, then collapse.
//!   Fast and often right, but it finds the most probable *alignment*, not
//!   the most probable *labeling*, and those can differ.
//! * [`prefix_beam_search`]: keep the best few labelings-so-far (prefixes)
//!   as we sweep over the frames, adding up the probability of all the
//!   alignments that lead to each one.
//! * [`lexicon_beam_search`]: the same search, but a prefix is only allowed
//!   to exist if it can still become a sequence of words from a given
//!   vocabulary.
//!
//! All of them take **log-probabilities**: a `Mat` with one row per frame and
//! one column per class, each row already passed through [`log_softmax`].
//! Scores inside the decoders are `f32` logarithms.

use crate::mat::Mat;
use std::collections::BTreeMap;

pub use crate::ctc::{collapse, log_softmax, BLANK};

/// The characters our recognizer can spell, and their class ids:
/// 0 is the CTC blank, 1 is the space between words, 2..=27 are `a`..=`z`.
///
/// This is the one place that mapping is written down; everything else goes
/// through [`Alphabet::encode`] and [`Alphabet::decode`].
pub struct Alphabet;

impl Alphabet {
    /// Class id of the CTC blank.
    pub const BLANK: usize = 0;
    /// Class id of the space character.
    pub const SPACE: usize = 1;
    /// Class id of the letter `a`; `b` is the next one, and so on.
    pub const FIRST_LETTER: usize = 2;
    /// Number of classes: blank + space + 26 letters.
    pub const SIZE: usize = 28;

    /// Text to class ids. Upper case is folded to lower case; anything that
    /// is not a letter `a`-`z` or a space (digits, punctuation) is dropped.
    /// The blank is never produced: it is not a character.
    pub fn encode(text: &str) -> Vec<usize> {
        let mut labels = Vec::with_capacity(text.len());
        for ch in text.chars() {
            let ch = ch.to_ascii_lowercase();
            if ch == ' ' {
                labels.push(Self::SPACE);
            } else if ch.is_ascii_lowercase() {
                labels.push(Self::FIRST_LETTER + (ch as usize - 'a' as usize));
            }
        }
        labels
    }

    /// Class ids to text. Blanks are skipped (a collapsed labeling has none
    /// anyway).
    pub fn decode(labels: &[usize]) -> String {
        labels.iter().filter(|&&label| label != Self::BLANK).map(|&label| Self::symbol(label)).collect()
    }

    /// The character to print for one class: `_` for the blank, a space for
    /// the space, the letter otherwise. Handy for printing alignments, where
    /// we want to see the blanks.
    pub fn symbol(class: usize) -> char {
        match class {
            Self::BLANK => '_',
            Self::SPACE => ' ',
            c if c < Self::SIZE => (b'a' + (c - Self::FIRST_LETTER) as u8) as char,
            _ => '?',
        }
    }
}

/// `ln(e^a + e^b)` for `f32` scores; see [`crate::ctc::log_add`].
fn log_add(a: f32, b: f32) -> f32 {
    crate::ctc::log_add(a as f64, b as f64) as f32
}

/// The most probable class in each frame: the single most probable
/// alignment. Ties go to the lower class id.
pub fn best_path(log_probs: &Mat) -> Vec<usize> {
    let mut path = Vec::with_capacity(log_probs.rows);
    for t in 0..log_probs.rows {
        let row = log_probs.row(t);
        let mut best = 0;
        for (class, &value) in row.iter().enumerate() {
            if value > row[best] {
                best = class;
            }
        }
        path.push(best);
    }
    path
}

/// Greedy decoding: [`best_path`], then [`collapse`].
pub fn greedy(log_probs: &Mat) -> Vec<usize> {
    collapse(&best_path(log_probs))
}

/// One decoded labeling and its score.
#[derive(Clone, Debug, PartialEq)]
pub struct Hypothesis {
    /// The class ids, collapsed (no blanks, repeats only where intended).
    pub labels: Vec<usize>,
    /// `ln p(labels | input)`: the log of the summed probability of the
    /// alignments the search kept for this labeling. With a wide enough beam
    /// that is all of them.
    pub log_prob: f32,
}

/// One prefix being tracked by the beam search, part-way through the input.
///
/// The probability of a prefix is kept as two separate numbers, because what
/// happens next depends on how the alignments so far *ended*. Take the prefix
/// `a` and suppose the next frame says `a` again:
///
/// * alignments that ended in `a` (`.. a`) just continue the same run:
///   `.. a a` still collapses to `a`;
/// * alignments that ended in a blank (`.. a _`) start a new letter:
///   `.. a _ a` collapses to `a a`.
///
/// So we need to know how much of the prefix's probability is of each kind.
#[derive(Clone, Debug, PartialEq)]
pub struct BeamEntry {
    /// The labeling so far.
    pub labels: Vec<usize>,
    /// Log of the summed probability of the alignments of `labels` that end
    /// in a blank.
    pub log_p_blank: f32,
    /// Log of the summed probability of the alignments of `labels` that end
    /// in its last label (not a blank).
    pub log_p_non_blank: f32,
    /// Lexicon decoding only: the trie node reached by the letters of the
    /// word currently being spelled (the root when there is no lexicon).
    node: usize,
    /// Lexicon decoding only: bonuses collected for the words completed so
    /// far (zero when there is no lexicon).
    word_score: f32,
}

impl BeamEntry {
    /// Log of the total probability of this prefix: blank-ending plus
    /// non-blank-ending alignments.
    pub fn log_total(&self) -> f32 {
        log_add(self.log_p_blank, self.log_p_non_blank)
    }

    /// What the beam is ranked by: the acoustic probability plus, in lexicon
    /// decoding, the word bonuses.
    fn score(&self) -> f32 {
        self.log_total() + self.word_score
    }
}

/// The beam before any frame has been seen: just the empty prefix, with
/// probability 1 (log 0), counted as "ending in a blank" so that any first
/// label starts a new run.
pub fn initial_beam() -> Vec<BeamEntry> {
    vec![BeamEntry { labels: Vec::new(), log_p_blank: 0.0, log_p_non_blank: f32::NEG_INFINITY, node: Lexicon::ROOT, word_score: 0.0 }]
}

/// Advance the beam by one frame: extend every prefix in every possible way,
/// merge the results that spell the same prefix, keep the best `beam_width`.
///
/// `frame` is one row of log-probabilities. Returns the new beam, best first.
pub fn beam_step(beam: &[BeamEntry], frame: &[f32], beam_width: usize) -> Vec<BeamEntry> {
    advance(beam, frame, beam_width, None)
}

/// Which of an entry's two probabilities a contribution belongs to.
#[derive(Clone, Copy)]
enum Ending {
    Blank,
    NonBlank,
}

/// Add `log_p` to the entry for `labels` in the next beam, creating the entry
/// if this is the first alignment to reach that prefix.
fn add_mass(next: &mut BTreeMap<Vec<usize>, BeamEntry>, labels: &[usize], node: usize, word_score: f32, ending: Ending, log_p: f32) {
    if log_p == f32::NEG_INFINITY {
        return; // nothing to add
    }
    let entry = next.entry(labels.to_vec()).or_insert_with(|| BeamEntry {
        labels: labels.to_vec(),
        log_p_blank: f32::NEG_INFINITY,
        log_p_non_blank: f32::NEG_INFINITY,
        node,
        word_score,
    });
    match ending {
        Ending::Blank => entry.log_p_blank = log_add(entry.log_p_blank, log_p),
        Ending::NonBlank => entry.log_p_non_blank = log_add(entry.log_p_non_blank, log_p),
    }
}

/// The step shared by the plain and the lexicon-constrained search.
///
/// For a prefix `l` with probabilities `p_blank` and `p_non_blank`, and a
/// frame with probabilities `y`, there are exactly three things an alignment
/// can do in this frame:
///
/// ```text
/// 1. emit a blank:                 l stays l, now ending in blank
///        next[l].p_blank     += (p_blank + p_non_blank) * y[blank]
///
/// 2. emit l's last label again:    l stays l, still ending in that label
///        next[l].p_non_blank += p_non_blank * y[last]
///
/// 3. emit a label c, growing l to l+c
///        next[l+c].p_non_blank += (p_blank + p_non_blank) * y[c]     if c != last
///        next[l+c].p_non_blank +=  p_blank                * y[c]     if c == last
/// ```
///
/// Case 3 with `c == last` only counts the blank-ending alignments: a repeat
/// straight after a non-blank is case 2, the same run continuing. The `+=`
/// is where the *sum over alignments* happens: several prefixes in the old
/// beam can land on the same new prefix, and their probabilities add.
fn advance(beam: &[BeamEntry], frame: &[f32], beam_width: usize, lexicon: Option<(&Lexicon, &LexiconOptions)>) -> Vec<BeamEntry> {
    assert!(beam_width > 0, "the beam must hold at least one prefix");
    let mut next: BTreeMap<Vec<usize>, BeamEntry> = BTreeMap::new();
    let mut extended = Vec::new();
    for entry in beam {
        let total = entry.log_total();
        let last = entry.labels.last().copied();

        // Case 1: a blank.
        add_mass(&mut next, &entry.labels, entry.node, entry.word_score, Ending::Blank, total + frame[BLANK]);

        // Case 2: the last label again, with no blank in between.
        if let Some(last) = last {
            add_mass(&mut next, &entry.labels, entry.node, entry.word_score, Ending::NonBlank, entry.log_p_non_blank + frame[last]);
        }

        // Case 3: grow the prefix by one label.
        for (class, &log_y) in frame.iter().enumerate().skip(1) {
            // With a lexicon, ask whether this letter keeps us inside the
            // vocabulary; if not, this extension is simply never created.
            let (node, word_score) = match lexicon {
                None => (Lexicon::ROOT, 0.0),
                Some((lexicon, options)) => match lexicon.step(entry.node, class, options) {
                    Some((node, bonus)) => (node, entry.word_score + bonus),
                    None => continue,
                },
            };
            let before = if Some(class) == last { entry.log_p_blank } else { total };
            extended.clear();
            extended.extend_from_slice(&entry.labels);
            extended.push(class);
            add_mass(&mut next, &extended, node, word_score, Ending::NonBlank, before + log_y);
        }
    }

    // Keep the best `beam_width` prefixes. The map hands them over in
    // dictionary order and the sort is stable, so ties are broken the same
    // way on every run.
    let mut ranked: Vec<BeamEntry> = next.into_values().collect();
    ranked.sort_by(|a, b| b.score().total_cmp(&a.score()));
    ranked.truncate(beam_width);
    ranked
}

/// CTC prefix beam search.
///
/// Sweeps over the frames once, keeping at most `beam_width` prefixes alive
/// (see [`beam_step`] for what happens at each frame). Returns the surviving
/// labelings, most probable first.
///
/// The search is exact if the beam never has to throw anything away, and an
/// approximation otherwise: a prefix that is dropped takes its probability
/// with it, even if a later frame would have made it the winner.
pub fn prefix_beam_search(log_probs: &Mat, beam_width: usize) -> Vec<Hypothesis> {
    let mut beam = initial_beam();
    for t in 0..log_probs.rows {
        beam = beam_step(&beam, log_probs.row(t), beam_width);
    }
    beam.into_iter().map(|entry| Hypothesis { log_prob: entry.log_total(), labels: entry.labels }).collect()
}

/// The exact probability of **every** labeling, by enumerating all `C^T`
/// alignments and adding each one's probability to the labeling it
/// collapses to. Most probable first.
///
/// This is what beam search approximates. It is only feasible for toy inputs
/// (a handful of frames and classes); the tests and the book's examples use
/// it as the ground truth.
pub fn all_labelings(log_probs: &Mat) -> Vec<Hypothesis> {
    let mut probability: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
    crate::ctc::for_each_alignment(log_probs.rows, log_probs.cols, |path| {
        let p = crate::ctc::alignment_log_prob(log_probs, path).exp();
        *probability.entry(collapse(path)).or_insert(0.0) += p;
    });
    let mut all: Vec<Hypothesis> = probability.into_iter().map(|(labels, p)| Hypothesis { labels, log_prob: p.ln() as f32 }).collect();
    all.sort_by(|a, b| b.log_prob.total_cmp(&a.log_prob));
    all
}

/// A vocabulary stored as a **trie**, a tree with one edge per letter.
///
/// Start at the root and follow the edges `c`, `a`, `t`: if you can, "cat" is
/// at least the beginning of a word, and the node you arrive at knows whether
/// it is a whole word. Words that share a beginning share the path: "three"
/// and "tree" both start by following `t`. Checking whether a prefix can
/// still become a word is therefore one array lookup per letter, no matter
/// how large the vocabulary is, which is exactly the question the beam search
/// asks thousands of times.
///
/// ```text
/// root ── t ── r ── e ── e*        * = a word ends here
///         │
///         └─── h ── r ── e ── e*
///         └─── w ── o*
/// ```
#[derive(Clone, Debug)]
pub struct Lexicon {
    nodes: Vec<TrieNode>,
    words: Vec<String>,
    /// One log-probability per word, same order as `words`.
    word_log_probs: Vec<f32>,
}

/// One node of the trie.
#[derive(Clone, Debug)]
struct TrieNode {
    /// `children[class]` is the node reached by that letter, or `NO_NODE`.
    children: [usize; Alphabet::SIZE],
    /// If the letters from the root to here spell a word, its index in
    /// `Lexicon::words`.
    word: Option<usize>,
}

/// Marks a missing edge in [`TrieNode::children`].
const NO_NODE: usize = usize::MAX;

impl Lexicon {
    /// The trie node every word starts from.
    const ROOT: usize = 0;

    /// Build a lexicon from a word list. Every word gets the same weight.
    /// Words must be non-empty and made of the letters `a`-`z` only.
    pub fn new(words: &[&str]) -> Self {
        let mut lexicon = Lexicon { nodes: vec![TrieNode { children: [NO_NODE; Alphabet::SIZE], word: None }], words: Vec::new(), word_log_probs: Vec::new() };
        for word in words {
            lexicon.insert(word);
        }
        lexicon
    }

    /// Build a lexicon in which each word comes with a log-probability, for
    /// example the log of how often it occurs (a **unigram** model: each
    /// word scored on its own, ignoring its neighbors).
    pub fn with_log_probs(words: &[(&str, f32)]) -> Self {
        let mut lexicon = Lexicon::new(&words.iter().map(|&(word, _)| word).collect::<Vec<_>>());
        // `insert` ignores repeated words, so look each one up again.
        for &(word, log_prob) in words {
            let index = lexicon.words.iter().position(|w| w == &word.to_ascii_lowercase()).expect("word was just inserted");
            lexicon.word_log_probs[index] = log_prob;
        }
        lexicon
    }

    /// Walk the word's letters down from the root, creating nodes as needed,
    /// and mark the last node as the end of a word.
    fn insert(&mut self, word: &str) {
        let letters = Alphabet::encode(word);
        assert!(!letters.is_empty(), "lexicon: empty word");
        assert!(letters.len() == word.chars().count() && !letters.contains(&Alphabet::SPACE), "lexicon: {word:?} must contain only the letters a-z");
        let mut node = Self::ROOT;
        for letter in letters {
            if self.nodes[node].children[letter] == NO_NODE {
                self.nodes.push(TrieNode { children: [NO_NODE; Alphabet::SIZE], word: None });
                self.nodes[node].children[letter] = self.nodes.len() - 1;
            }
            node = self.nodes[node].children[letter];
        }
        if self.nodes[node].word.is_none() {
            self.nodes[node].word = Some(self.words.len());
            self.words.push(word.to_ascii_lowercase());
            self.word_log_probs.push(0.0);
        }
    }

    /// The words in the lexicon, in the order they were first added.
    pub fn words(&self) -> &[String] {
        &self.words
    }

    /// Is this exact word in the lexicon?
    pub fn contains(&self, word: &str) -> bool {
        let mut node = Self::ROOT;
        for letter in Alphabet::encode(word) {
            match self.child(node, letter) {
                Some(child) => node = child,
                None => return false,
            }
        }
        self.nodes[node].word.is_some()
    }

    /// The node reached from `node` by one letter, if that edge exists.
    fn child(&self, node: usize, class: usize) -> Option<usize> {
        match self.nodes[node].children.get(class) {
            Some(&child) if child != NO_NODE => Some(child),
            _ => None,
        }
    }

    /// The score a word earns when it is completed: the flat per-word bonus
    /// plus its weighted unigram log-probability.
    fn word_reward(&self, word: usize, options: &LexiconOptions) -> f32 {
        options.word_bonus + options.unigram_weight * self.word_log_probs[word]
    }

    /// Try to extend a prefix whose current (partial) word sits at `node`
    /// with the label `class`. Returns the new node and the score earned by
    /// the step, or `None` if the vocabulary does not allow it.
    ///
    /// * A letter is allowed if the trie has that edge.
    /// * A space is allowed only when the letters so far spell a whole word.
    ///   It sends us back to the root, ready for the next word, and pays out
    ///   that word's reward. A space at the root (at the very start, or
    ///   straight after another space) is never allowed.
    fn step(&self, node: usize, class: usize, options: &LexiconOptions) -> Option<(usize, f32)> {
        if class == Alphabet::SPACE {
            let word = self.nodes[node].word?;
            Some((Self::ROOT, self.word_reward(word, options)))
        } else {
            self.child(node, class).map(|child| (child, 0.0))
        }
    }
}

/// Settings for [`lexicon_beam_search`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LexiconOptions {
    /// How many prefixes to keep after each frame.
    pub beam_width: usize,
    /// Added to a hypothesis' score once per word it contains. Positive
    /// values encourage more, shorter words; negative values (a word
    /// insertion penalty) discourage them. In log units.
    pub word_bonus: f32,
    /// How much the lexicon's per-word log-probabilities count. 0.0 ignores
    /// them; 1.0 adds them as they are.
    pub unigram_weight: f32,
}

impl LexiconOptions {
    /// A beam of the given width with no word bonus and no unigram scores:
    /// the lexicon only says which spellings are allowed.
    pub fn new(beam_width: usize) -> Self {
        LexiconOptions { beam_width, word_bonus: 0.0, unigram_weight: 0.0 }
    }
}

/// A decoded sequence of vocabulary words.
#[derive(Clone, Debug, PartialEq)]
pub struct WordHypothesis {
    /// The words, in order.
    pub words: Vec<String>,
    /// The spelled-out class ids (letters and single spaces).
    pub labels: Vec<usize>,
    /// `ln p(labels | input)` from the network alone, as in [`Hypothesis`].
    pub log_prob: f32,
    /// `log_prob` plus the word bonuses and unigram scores: what the results
    /// are ranked by.
    pub score: f32,
}

impl WordHypothesis {
    /// The words joined by spaces.
    pub fn text(&self) -> String {
        self.words.join(" ")
    }
}

/// Prefix beam search that can only spell sequences of vocabulary words.
///
/// It is [`prefix_beam_search`] with one extra rule: a prefix may be extended
/// by a label only if [`Lexicon`] allows it, that is, if the result is still
/// zero or more whole words separated by single spaces, followed by the
/// beginning of a word. Each prefix remembers which trie node its unfinished
/// word has reached, so the check costs one lookup.
///
/// The network may put a little probability on "lefd"; the plain search would
/// happily return it if it scored best. Here the `d` is never tried after
/// `lef`, because no word continues that way, and the probability goes to
/// the spellings that *are* words.
///
/// Scoring: a hypothesis' score is its CTC log-probability, plus
/// `options.word_bonus` per word, plus `options.unigram_weight` times each
/// word's log-probability from the lexicon. The reward for a word is added
/// when the space after it is emitted (so it can influence which prefixes
/// survive), and for the final word at the very end.
///
/// Returns only complete hypotheses (the last word is a whole word, with no
/// space after it), best first. The list is empty if no complete hypothesis
/// survived in the beam.
pub fn lexicon_beam_search(log_probs: &Mat, lexicon: &Lexicon, options: &LexiconOptions) -> Vec<WordHypothesis> {
    let mut beam = initial_beam();
    for t in 0..log_probs.rows {
        beam = advance(&beam, log_probs.row(t), options.beam_width, Some((lexicon, options)));
    }

    let mut finished = Vec::new();
    for entry in beam {
        // Complete means: nothing at all, or the unfinished word is in fact
        // a whole word. (At the root with labels behind us, the prefix ends
        // in a space and is waiting for a word that never came.)
        let final_reward = match lexicon.nodes[entry.node].word {
            Some(word) => lexicon.word_reward(word, options),
            None if entry.labels.is_empty() => 0.0,
            None => continue,
        };
        let text = Alphabet::decode(&entry.labels);
        finished.push(WordHypothesis {
            words: text.split(' ').filter(|w| !w.is_empty()).map(String::from).collect(),
            log_prob: entry.log_total(),
            score: entry.score() + final_reward,
            labels: entry.labels,
        });
    }
    finished.sort_by(|a, b| b.score.total_cmp(&a.score));
    finished
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Log-probabilities from a table of probabilities, one row per frame.
    fn table(rows: &[&[f32]]) -> Mat {
        Mat::from_rows(&rows.iter().map(|row| row.iter().map(|p| p.ln()).collect()).collect::<Vec<_>>())
    }

    fn random_log_probs(rng: &mut Rng, frames: usize, classes: usize) -> Mat {
        log_softmax(&Mat::from_fn(frames, classes, |_, _| 1.5 * rng.normal()))
    }

    #[test]
    fn alphabet_round_trips() {
        assert_eq!(Alphabet::encode("a z"), vec![2, 1, 27]);
        assert_eq!(Alphabet::encode("Go Left!"), Alphabet::encode("go left"));
        assert_eq!(Alphabet::decode(&Alphabet::encode("three two one")), "three two one");
        assert_eq!(Alphabet::decode(&[0, 4, 0, 2, 21, 0]), "cat");
        assert_eq!(Alphabet::symbol(0), '_');
        assert_eq!(Alphabet::encode("abcdefghijklmnopqrstuvwxyz"), (2..=27).collect::<Vec<_>>());
    }

    #[test]
    fn greedy_takes_the_argmax_and_collapses() {
        // Frames vote: a, a, blank, a, b, b  ->  "aab"
        let lp = table(&[&[0.1, 0.8, 0.1], &[0.2, 0.7, 0.1], &[0.9, 0.05, 0.05], &[0.1, 0.6, 0.3], &[0.1, 0.2, 0.7], &[0.3, 0.3, 0.4]]);
        assert_eq!(best_path(&lp), vec![1, 1, 0, 1, 2, 2]);
        assert_eq!(greedy(&lp), vec![1, 1, 2]);
        assert!(greedy(&Mat::zeros(0, 3)).is_empty());
    }

    /// The smallest case where greedy is wrong. Two frames, classes
    /// {blank, a}; in both frames blank has 0.6 and `a` has 0.4.
    ///
    ///   greedy: blank, blank -> ""        p("")  = 0.6 * 0.6          = 0.36
    ///   but     "a" = a_ + _a + aa        p("a") = 0.24 + 0.24 + 0.16 = 0.64
    ///
    /// The blank wins each frame on its own, yet "a" is almost twice as
    /// likely overall, because three different alignments spell it.
    #[test]
    fn greedy_fails_where_beam_search_succeeds_minimal() {
        let lp = table(&[&[0.6, 0.4], &[0.6, 0.4]]);
        assert_eq!(greedy(&lp), Vec::<usize>::new());
        let beam = prefix_beam_search(&lp, 4);
        assert_eq!(beam[0].labels, vec![1]);
        assert!((beam[0].log_prob.exp() - 0.64).abs() < 1e-6);
        assert_eq!(beam[1].labels, Vec::<usize>::new());
        assert!((beam[1].log_prob.exp() - 0.36).abs() < 1e-6);
    }

    /// The example the book walks through. Three frames, classes {blank, a, b}:
    ///
    ///            blank    a     b
    ///   frame 0   0.5    0.4   0.1
    ///   frame 1   0.5    0.4   0.1
    ///   frame 2   0.3    0.1   0.6
    ///
    /// Greedy picks blank, blank, b -> "b". Summing alignments instead:
    ///
    ///   "b":  __b .150  _bb .030  bbb .006  _b_ .015  b__ .015  bb_ .003   = 0.219
    ///   "ab": a_b .120  _ab .120  aab .096  abb .024  ab_ .012             = 0.372
    ///
    /// The `a` never wins a frame, because the evidence for it is split
    /// between "it happened in frame 0" and "it happened in frame 1". Greedy
    /// looks at one frame at a time and drops the letter; the beam search
    /// adds the two possibilities up and keeps it.
    #[test]
    fn greedy_fails_where_beam_search_succeeds_book_example() {
        let lp = table(&[&[0.5, 0.4, 0.1], &[0.5, 0.4, 0.1], &[0.3, 0.1, 0.6]]);
        assert_eq!(best_path(&lp), vec![0, 0, 2]);
        assert_eq!(greedy(&lp), vec![2]);

        let exact = all_labelings(&lp);
        assert_eq!(exact[0].labels, vec![1, 2]);
        assert!((exact[0].log_prob.exp() - 0.372).abs() < 1e-6);
        let b = exact.iter().find(|h| h.labels == [2]).unwrap();
        assert!((b.log_prob.exp() - 0.219).abs() < 1e-6);

        // A beam wide enough to keep every prefix gets the exact probability.
        for width in [10, 100] {
            let beam = prefix_beam_search(&lp, width);
            assert_eq!(beam[0].labels, vec![1, 2], "width {width}");
            assert!((beam[0].log_prob.exp() - 0.372).abs() < 1e-5, "width {width}");
        }
        // A beam of 3 still finds "ab", but underestimates it: after frame 1
        // the prefix "ab" (alignment `a b`, 0.04) ranks fourth and is thrown
        // away, and with it the alignments abb (0.024) and ab_ (0.012).
        let narrow = prefix_beam_search(&lp, 3);
        assert_eq!(narrow[0].labels, vec![1, 2]);
        assert!((narrow[0].log_prob.exp() - 0.336).abs() < 1e-5);
        // A beam of 1 is no better than greedy here.
        assert_eq!(prefix_beam_search(&lp, 1)[0].labels, vec![2]);
    }

    /// With a beam wide enough to hold every prefix, beam search is exact:
    /// same labelings, same probabilities as the enumeration.
    #[test]
    fn wide_beam_matches_brute_force_exactly() {
        let mut rng = Rng::new(31);
        for (frames, classes) in [(1, 3), (4, 3), (5, 3), (6, 3), (4, 4), (5, 4)] {
            let lp = random_log_probs(&mut rng, frames, classes);
            let exact = all_labelings(&lp);
            let beam = prefix_beam_search(&lp, 100_000);
            assert_eq!(beam.len(), exact.len(), "T={frames} C={classes}");
            assert_eq!(beam[0].labels, exact[0].labels);
            for hypothesis in &beam {
                let truth = exact.iter().find(|h| h.labels == hypothesis.labels).unwrap();
                assert!((hypothesis.log_prob - truth.log_prob).abs() < 1e-4, "{:?}: {} vs {}", hypothesis.labels, hypothesis.log_prob, truth.log_prob);
            }
            assert!(beam.windows(2).all(|w| w[0].log_prob >= w[1].log_prob), "not sorted");
            let total: f32 = beam.iter().map(|h| h.log_prob.exp()).sum();
            assert!((total - 1.0).abs() < 1e-4);
        }
    }

    /// A narrow beam can lose probability but can never invent any.
    #[test]
    fn narrow_beam_never_overestimates() {
        let mut rng = Rng::new(32);
        let lp = random_log_probs(&mut rng, 6, 4);
        let exact = all_labelings(&lp);
        for width in [1, 2, 4, 8] {
            let beam = prefix_beam_search(&lp, width);
            assert!(beam.len() <= width);
            for hypothesis in &beam {
                let truth = exact.iter().find(|h| h.labels == hypothesis.labels).unwrap();
                assert!(hypothesis.log_prob <= truth.log_prob + 1e-4);
            }
        }
    }

    #[test]
    fn beam_trace_can_be_followed_step_by_step() {
        let lp = table(&[&[0.5, 0.4, 0.1], &[0.5, 0.4, 0.1]]);
        let beam = beam_step(&initial_beam(), lp.row(0), 10);
        // After one frame: "" (0.5, ends in blank), "a" (0.4), "b" (0.1).
        assert_eq!(beam.iter().map(|e| e.labels.clone()).collect::<Vec<_>>(), vec![vec![], vec![1], vec![2]]);
        assert_eq!(beam[0].log_p_non_blank, f32::NEG_INFINITY);
        assert_eq!(beam[1].log_p_blank, f32::NEG_INFINITY);
        let beam = beam_step(&beam, lp.row(1), 10);
        // "a" after two frames: a_ (0.4*0.5, ends in blank), _a (0.5*0.4) and aa (0.4*0.4).
        let a = beam.iter().find(|e| e.labels == [1]).unwrap();
        assert!((a.log_p_blank.exp() - 0.20).abs() < 1e-6);
        assert!((a.log_p_non_blank.exp() - 0.36).abs() < 1e-6);
        // "aa" needs a blank between the two a's, impossible in two frames.
        assert!(beam.iter().all(|e| e.labels != [1, 1]));
    }

    #[test]
    fn trie_knows_words_and_prefixes() {
        let lexicon = Lexicon::new(&["tree", "three", "two", "tree"]);
        assert_eq!(lexicon.words(), ["tree", "three", "two"]);
        assert!(lexicon.contains("tree") && lexicon.contains("three") && lexicon.contains("Two"));
        assert!(!lexicon.contains("tr") && !lexicon.contains("trees") && !lexicon.contains(""));
        // t, then (r | h | w): shared first letter, one root edge.
        let root_edges = lexicon.nodes[0].children.iter().filter(|&&c| c != NO_NODE).count();
        assert_eq!(root_edges, 1);
        // root + t + (r,e,e) + (h,r,e,e) + (w,o) = 11 nodes.
        assert_eq!(lexicon.nodes.len(), 11);
    }

    /// Is this label sequence zero or more lexicon words separated by single spaces?
    fn is_word_sequence(lexicon: &Lexicon, labels: &[usize]) -> bool {
        labels.is_empty() || Alphabet::decode(labels).split(' ').all(|word| lexicon.contains(word))
    }

    /// With a wide beam the constrained search must return exactly the valid
    /// labelings, each with its exact probability, and so its top answer is
    /// the most probable *valid* labeling. Classes: blank, space, a, b.
    #[test]
    fn lexicon_search_matches_brute_force_over_valid_labelings() {
        let mut rng = Rng::new(33);
        let lexicon = Lexicon::new(&["a", "ab", "ba", "bb"]);
        for frames in [1, 3, 5, 6] {
            let lp = random_log_probs(&mut rng, frames, 4);
            let valid: Vec<Hypothesis> = all_labelings(&lp).into_iter().filter(|h| is_word_sequence(&lexicon, &h.labels)).collect();
            let found = lexicon_beam_search(&lp, &lexicon, &LexiconOptions::new(100_000));
            assert_eq!(found.len(), valid.len(), "T={frames}");
            assert_eq!(found[0].labels, valid[0].labels);
            for hypothesis in &found {
                let truth = valid.iter().find(|h| h.labels == hypothesis.labels).unwrap();
                assert!((hypothesis.log_prob - truth.log_prob).abs() < 1e-4);
                assert_eq!(hypothesis.score, hypothesis.log_prob);
                assert_eq!(Alphabet::encode(&hypothesis.text()), hypothesis.labels);
            }
        }
    }

    /// The network slightly prefers the non-word "lefd"; the lexicon rules it out.
    #[test]
    fn lexicon_fixes_a_misspelling() {
        let mut lp = Mat::from_fn(8, Alphabet::SIZE, |_, _| 0.002f32);
        let spelled = Alphabet::encode("left");
        for (i, &letter) in spelled.iter().enumerate() {
            lp[(2 * i, letter)] = 0.9;
            lp[(2 * i + 1, Alphabet::BLANK)] = 0.9;
        }
        // Frame 6 hesitates between t and d, leaning the wrong way.
        lp[(6, Alphabet::encode("t")[0])] = 0.42;
        lp[(6, Alphabet::encode("d")[0])] = 0.50;
        let lp = lp.map(f32::ln);

        assert_eq!(Alphabet::decode(&greedy(&lp)), "lefd");
        assert_eq!(Alphabet::decode(&prefix_beam_search(&lp, 16)[0].labels), "lefd");
        let lexicon = Lexicon::new(&["left", "let", "go", "off"]);
        let found = lexicon_beam_search(&lp, &lexicon, &LexiconOptions::new(16));
        assert_eq!(found[0].text(), "left");
        assert_eq!(found[0].words, vec!["left"]);
        assert!(found.iter().all(|h| is_word_sequence(&lexicon, &h.labels)));
    }

    /// Three frames: a, then space-or-blank (blank slightly ahead), then b.
    /// "ab" is one word, "a b" is two.
    #[test]
    fn word_bonus_and_unigrams_change_the_ranking() {
        let mut probs = Mat::from_fn(3, 4, |_, _| 0.01f32);
        probs[(0, 2)] = 0.97;
        probs[(1, Alphabet::BLANK)] = 0.54;
        probs[(1, Alphabet::SPACE)] = 0.44;
        probs[(2, 3)] = 0.97;
        let lp = probs.map(f32::ln);
        let lexicon = Lexicon::new(&["a", "b", "ab"]);

        let plain = lexicon_beam_search(&lp, &lexicon, &LexiconOptions::new(50));
        assert_eq!(plain[0].text(), "ab");

        // A bonus per word favors the two-word reading: it earns it twice.
        let bonus = LexiconOptions { word_bonus: 1.0, ..LexiconOptions::new(50) };
        let with_bonus = lexicon_beam_search(&lp, &lexicon, &bonus);
        assert_eq!(with_bonus[0].text(), "a b");
        assert!((with_bonus[0].score - (with_bonus[0].log_prob + 2.0)).abs() < 1e-5);

        // A penalty per word leaves "ab" in front.
        let penalty = LexiconOptions { word_bonus: -1.0, ..LexiconOptions::new(50) };
        assert_eq!(lexicon_beam_search(&lp, &lexicon, &penalty)[0].text(), "ab");

        // Unigram scores: make "ab" a rare word and "a", "b" common ones.
        let weighted = Lexicon::with_log_probs(&[("a", 0.4f32.ln()), ("b", 0.4f32.ln()), ("ab", 0.01f32.ln())]);
        let unigram = LexiconOptions { unigram_weight: 1.0, ..LexiconOptions::new(50) };
        let with_unigrams = lexicon_beam_search(&lp, &weighted, &unigram);
        assert_eq!(with_unigrams[0].text(), "a b");
        let expected = with_unigrams[0].log_prob + 2.0 * 0.4f32.ln();
        assert!((with_unigrams[0].score - expected).abs() < 1e-5);
        // With weight 0 the same lexicon behaves like the unweighted one.
        assert_eq!(lexicon_beam_search(&lp, &weighted, &LexiconOptions::new(50))[0].text(), "ab");
    }

    #[test]
    fn empty_input_decodes_to_nothing() {
        let lp = Mat::zeros(0, Alphabet::SIZE);
        assert_eq!(prefix_beam_search(&lp, 8), vec![Hypothesis { labels: vec![], log_prob: 0.0 }]);
        let found = lexicon_beam_search(&lp, &Lexicon::new(&["yes"]), &LexiconOptions::new(8));
        assert_eq!(found.len(), 1);
        assert!(found[0].words.is_empty());
    }
}
