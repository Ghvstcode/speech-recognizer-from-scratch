//! Turning token ids back into text.
//!
//! Whisper does not spell words letter by letter the way our own recognizer
//! does. Its output alphabet is about 50,000 *tokens*: common words (` the`),
//! pieces of words (`ing`), and, as a fallback, single bytes. The scheme is
//! byte-pair encoding (BPE), the same vocabulary GPT-2 uses. *Encoding* text
//! into tokens takes some machinery; *decoding*, all we need, is a table
//! lookup: every token stands for a fixed string of bytes, so we look up
//! each id, glue the bytes together, and read the result as UTF-8.
//!
//! One wrinkle. A token is a string of arbitrary *bytes* (a token may even
//! be half of a multi-byte character), but `vocab.json` is a text file and
//! cannot hold arbitrary bytes. So GPT-2's authors gave each of the 256 byte
//! values a printable stand-in character:
//!
//! - bytes that are already printable (`!` to `~`, and most of Latin-1 from
//!   `¡` to `ÿ`) stand for themselves;
//! - the other 68 (control characters, the space, and a few more) are
//!   assigned, in increasing order, the characters U+0100, U+0101, ...
//!
//! The space (byte 32) is the 33rd unprintable byte and so becomes U+0120,
//! `Ġ`. That is why the vocabulary file is full of entries like `Ġthe`: it
//! means " the", with the space in front.

use super::json::Json;
use std::collections::HashMap;
use std::io;
use std::path::Path;

fn bad(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("tokenizer: {message}"))
}

/// Does this byte stand for itself in the vocabulary file?
fn is_printable(byte: u8) -> bool {
    matches!(byte, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF)
}

/// The stand-in character for each of the 256 byte values.
pub fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next_stand_in = 0x100;
    for byte in 0..=255u8 {
        table[byte as usize] = if is_printable(byte) {
            byte as char // Latin-1: the character's code point equals the byte
        } else {
            next_stand_in += 1;
            char::from_u32(next_stand_in - 1).expect("U+0100..U+0143 are valid characters")
        };
    }
    table
}

/// Maps token ids to the bytes they stand for.
pub struct Tokenizer {
    /// `pieces[id]` is the byte string of token `id`.
    pieces: Vec<Vec<u8>>,
    /// Ids from here upwards are control tokens (`<|endoftext|>`,
    /// `<|startoftranscript|>`, timestamps...), which are not part of the text.
    first_special: usize,
}

impl Tokenizer {
    /// Build from a parsed `vocab.json` (an object mapping token string to
    /// id) and, optionally, a parsed `added_tokens.json` of the same shape
    /// that names the control tokens. `first_special` is the id of
    /// `<|endoftext|>`, the lowest-numbered control token.
    pub fn from_json(vocab: &Json, added_tokens: Option<&Json>, first_special: usize) -> io::Result<Tokenizer> {
        // Invert the byte-to-character table.
        let char_to_byte: HashMap<char, u8> = byte_to_char_table().iter().enumerate().map(|(byte, &ch)| (ch, byte as u8)).collect();

        let mut pieces: Vec<Vec<u8>> = Vec::new();
        let mut store = |id: usize, bytes: Vec<u8>| {
            if pieces.len() <= id {
                pieces.resize(id + 1, Vec::new());
            }
            pieces[id] = bytes;
        };

        let entries = vocab.as_object().ok_or_else(|| bad("vocab.json is not an object".into()))?;
        for (text, id) in entries {
            let id = id.as_usize().ok_or_else(|| bad(format!("token {text:?} has no integer id")))?;
            let bytes: Option<Vec<u8>> = text.chars().map(|ch| char_to_byte.get(&ch).copied()).collect();
            store(id, bytes.ok_or_else(|| bad(format!("token {text:?} contains a character outside the byte alphabet")))?);
        }
        // Control tokens are written literally, e.g. "<|notimestamps|>".
        for (text, id) in added_tokens.and_then(Json::as_object).unwrap_or(&[]) {
            let id = id.as_usize().ok_or_else(|| bad(format!("token {text:?} has no integer id")))?;
            store(id, text.as_bytes().to_vec());
        }
        Ok(Tokenizer { pieces, first_special })
    }

    /// Load `vocab.json` and (if present) `added_tokens.json` from a model directory.
    pub fn load(dir: &Path, first_special: usize) -> io::Result<Tokenizer> {
        let vocab = Json::read_file(&dir.join("vocab.json"))?;
        let added_path = dir.join("added_tokens.json");
        let added = if added_path.exists() { Some(Json::read_file(&added_path)?) } else { None };
        Tokenizer::from_json(&vocab, added.as_ref(), first_special)
    }

    /// Number of token ids the tokenizer knows.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    /// True if the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Whether `id` is a control token rather than text.
    pub fn is_special(&self, id: usize) -> bool {
        id >= self.first_special
    }

    /// The bytes token `id` stands for (empty for an unknown id).
    pub fn piece(&self, id: usize) -> &[u8] {
        self.pieces.get(id).map_or(&[], Vec::as_slice)
    }

    /// One token as displayable text, for printing candidates. A token that
    /// is only part of a UTF-8 character shows as the replacement character.
    pub fn piece_text(&self, id: usize) -> String {
        String::from_utf8_lossy(self.piece(id)).into_owned()
    }

    /// The text a sequence of token ids stands for, control tokens skipped.
    ///
    /// The bytes of all tokens are concatenated *before* decoding as UTF-8,
    /// because a character such as `é` (two bytes) may be split across two
    /// tokens, and neither half is valid on its own.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if !self.is_special(id) {
                bytes.extend_from_slice(self.piece(id));
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_table_matches_gpt2s_construction() {
        let table = byte_to_char_table();
        // Printable bytes are themselves.
        assert_eq!(table[b'A' as usize], 'A');
        assert_eq!(table[b'~' as usize], '~');
        assert_eq!(table[0xE9], 'é');
        // The 33 bytes 0..=32 take U+0100..=U+0120, so space is Ġ.
        assert_eq!(table[0], '\u{100}');
        assert_eq!(table[b' ' as usize], 'Ġ');
        assert_eq!(table[b'\n' as usize], 'Ċ');
        // Then 127..=160 take U+0121..=U+0142, and the soft hyphen 173 is last.
        assert_eq!(table[127], '\u{121}');
        assert_eq!(table[160], '\u{142}');
        assert_eq!(table[173], '\u{143}');
        // All 256 stand-ins are distinct, so the mapping can be inverted.
        let distinct: std::collections::HashSet<char> = table.iter().copied().collect();
        assert_eq!(distinct.len(), 256);
    }

    fn toy() -> Tokenizer {
        // "é" is bytes C3 A9, which the vocabulary file shows as "Ã©".
        let vocab = Json::parse(r#"{"Hello": 0, "Ġworld": 1, "!": 2, "Ã": 3, "©": 4, "Ġcaf": 5, "<|endoftext|>": 6}"#).unwrap();
        let added = Json::parse(r#"{"<|startoftranscript|>": 7, "<|notimestamps|>": 8}"#).unwrap();
        Tokenizer::from_json(&vocab, Some(&added), 6).unwrap()
    }

    #[test]
    fn decodes_spaces_and_skips_control_tokens() {
        let tokenizer = toy();
        assert_eq!(tokenizer.len(), 9);
        assert_eq!(tokenizer.decode(&[7, 8, 0, 1, 2, 6]), "Hello world!");
        assert_eq!(tokenizer.piece(1), b" world");
        assert!(tokenizer.is_special(6) && !tokenizer.is_special(5));
        assert_eq!(tokenizer.piece_text(8), "<|notimestamps|>");
    }

    #[test]
    fn joins_bytes_across_tokens_before_decoding_utf8() {
        let tokenizer = toy();
        assert_eq!(tokenizer.piece(3), [0xC3]);
        assert_eq!(tokenizer.decode(&[5, 3, 4]), " café");
        // Half a character on its own is not text.
        assert_eq!(tokenizer.piece_text(3), "\u{FFFD}");
    }

    #[test]
    fn rejects_characters_outside_the_byte_alphabet() {
        let vocab = Json::parse(r#"{"世": 0}"#).unwrap();
        assert!(Tokenizer::from_json(&vocab, None, 1).is_err());
    }
}
