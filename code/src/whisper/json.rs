//! A small JSON parser.
//!
//! Whisper's files lean on JSON three times over: the header of the weights
//! file is JSON, the vocabulary is one big JSON object, and so are the
//! configuration files. JSON is small enough to parse by hand with the
//! classic technique called *recursive descent*: one function per kind of
//! value, each of which looks at the next character, decides what it is
//! looking at, and calls the others for anything nested inside.

use std::io;
use std::path::Path;

/// Any JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    /// `null`
    Null,
    /// `true` or `false`
    Bool(bool),
    /// Every JSON number, integer or not, is kept as an `f64`. That is exact
    /// for integers up to 2^53, far more than any offset or token id we meet.
    Number(f64),
    /// A string, with all escapes already resolved.
    String(String),
    /// `[a, b, c]`
    Array(Vec<Json>),
    /// `{"key": value, ...}` as a list of pairs **in file order**. A list
    /// rather than a hash map because order is occasionally interesting (the
    /// tensor listing prints in file order) and lookups in small objects are
    /// just as quick by scanning.
    Object(Vec<(String, Json)>),
}

/// Deeper nesting than this is refused, so a hostile file made of a million
/// `[` cannot overflow our stack.
const MAX_DEPTH: usize = 128;

impl Json {
    /// Parse a complete JSON document.
    pub fn parse(text: &str) -> io::Result<Json> {
        let mut parser = Parser { bytes: text.as_bytes(), pos: 0 };
        let value = parser.value(0)?;
        parser.skip_whitespace();
        if parser.pos != parser.bytes.len() {
            return Err(parser.error("unexpected text after the end of the document"));
        }
        Ok(value)
    }

    /// Read and parse a JSON file.
    pub fn read_file(path: &Path) -> io::Result<Json> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
        Json::parse(&text)
    }

    /// Look up `key` in an object. `None` if this is not an object or the key is absent.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// The number, if this is a number.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// The number as a `usize`, if this is a non-negative whole number.
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Number(n) if *n >= 0.0 && n.fract() == 0.0 => Some(*n as usize),
            _ => None,
        }
    }

    /// The text, if this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    /// The boolean, if this is `true` or `false`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The elements, if this is an array.
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The key/value pairs in file order, if this is an object.
    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(pairs) => Some(pairs),
            _ => None,
        }
    }
}

/// The parser's whole state: the input and how far into it we are.
struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn error(&self, message: &str) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, format!("JSON: {message} at byte {}", self.pos))
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Some(byte)
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, wanted: u8) -> io::Result<()> {
        if self.next() == Some(wanted) {
            Ok(())
        } else {
            self.pos = self.pos.saturating_sub(1);
            Err(self.error(&format!("expected '{}'", wanted as char)))
        }
    }

    /// Parse whatever value starts here. The first character decides which.
    fn value(&mut self, depth: usize) -> io::Result<Json> {
        if depth > MAX_DEPTH {
            return Err(self.error("nesting too deep"));
        }
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.error("unexpected character")),
            None => Err(self.error("unexpected end of input")),
        }
    }

    /// `true`, `false` and `null` are fixed words.
    fn literal(&mut self, word: &str, value: Json) -> io::Result<Json> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.error(&format!("expected `{word}`")))
        }
    }

    /// Numbers: gather every character that can appear in one and let the
    /// standard library do the conversion. (This accepts a few spellings that
    /// strict JSON forbids, such as `1.`; being lenient costs us nothing.)
    fn number(&mut self) -> io::Result<Json> {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')) {
            self.pos += 1;
        }
        // Only ASCII was consumed, so this slice is valid UTF-8.
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or("");
        match text.parse::<f64>() {
            Ok(n) => Ok(Json::Number(n)),
            Err(_) => {
                self.pos = start;
                Err(self.error("malformed number"))
            }
        }
    }

    /// `[ value, value, ... ]`
    fn array(&mut self, depth: usize) -> io::Result<Json> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_whitespace();
            match self.next() {
                Some(b',') => continue,
                Some(b']') => return Ok(Json::Array(items)),
                _ => return Err(self.error("expected ',' or ']' in array")),
            }
        }
    }

    /// `{ "key": value, ... }`
    fn object(&mut self, depth: usize) -> io::Result<Json> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(pairs));
        }
        loop {
            self.skip_whitespace();
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            let value = self.value(depth + 1)?;
            pairs.push((key, value));
            self.skip_whitespace();
            match self.next() {
                Some(b',') => continue,
                Some(b'}') => return Ok(Json::Object(pairs)),
                _ => return Err(self.error("expected ',' or '}' in object")),
            }
        }
    }

    /// A string: everything up to the closing quote, with backslash escapes
    /// replaced by the characters they stand for.
    ///
    /// We collect raw bytes rather than `char`s. The input is already valid
    /// UTF-8, a multi-byte character can never contain the bytes `"` or `\`,
    /// so copying bytes through untouched keeps every character intact.
    fn string(&mut self) -> io::Result<String> {
        self.expect(b'"')?;
        let mut out: Vec<u8> = Vec::new();
        loop {
            match self.next() {
                None => return Err(self.error("unterminated string")),
                Some(b'"') => break,
                Some(b'\\') => {
                    let ch = self.escape()?;
                    let mut utf8 = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut utf8).as_bytes());
                }
                Some(0..=0x1f) => return Err(self.error("raw control character in string")),
                Some(byte) => out.push(byte),
            }
        }
        String::from_utf8(out).map_err(|_| self.error("string is not valid UTF-8"))
    }

    /// The character named by the escape sequence after a backslash.
    fn escape(&mut self) -> io::Result<char> {
        let ch = match self.next() {
            Some(b'"') => '"',
            Some(b'\\') => '\\',
            Some(b'/') => '/',
            Some(b'b') => '\u{0008}',
            Some(b'f') => '\u{000C}',
            Some(b'n') => '\n',
            Some(b'r') => '\r',
            Some(b't') => '\t',
            Some(b'u') => return self.unicode_escape(),
            _ => return Err(self.error("unknown escape sequence")),
        };
        Ok(ch)
    }

    /// `\uXXXX`. JSON escapes are 16 bits wide, so a character beyond U+FFFF
    /// (an emoji, say) is written as *two* escapes, a "surrogate pair":
    /// a high half in D800..DBFF followed by a low half in DC00..DFFF, each
    /// carrying ten bits of the real code point.
    fn unicode_escape(&mut self) -> io::Result<char> {
        let first = self.hex4()?;
        let code_point = match first {
            0xD800..=0xDBFF => {
                if self.next() != Some(b'\\') || self.next() != Some(b'u') {
                    return Err(self.error("high surrogate not followed by \\u"));
                }
                let second = self.hex4()?;
                if !(0xDC00..=0xDFFF).contains(&second) {
                    return Err(self.error("high surrogate not followed by a low surrogate"));
                }
                0x10000 + ((first - 0xD800) << 10) + (second - 0xDC00)
            }
            0xDC00..=0xDFFF => return Err(self.error("low surrogate without a high surrogate")),
            other => other,
        };
        char::from_u32(code_point).ok_or_else(|| self.error("escape is not a valid character"))
    }

    /// Four hexadecimal digits as a number.
    fn hex4(&mut self) -> io::Result<u32> {
        let mut value = 0;
        for _ in 0..4 {
            let digit = self.next().and_then(|b| (b as char).to_digit(16));
            value = value * 16 + digit.ok_or_else(|| self.error("expected four hex digits after \\u"))?;
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars() {
        assert_eq!(Json::parse("null").unwrap(), Json::Null);
        assert_eq!(Json::parse(" true ").unwrap(), Json::Bool(true));
        assert_eq!(Json::parse("false").unwrap(), Json::Bool(false));
        assert_eq!(Json::parse("42").unwrap(), Json::Number(42.0));
        assert_eq!(Json::parse("-1.5e3").unwrap(), Json::Number(-1500.0));
        assert_eq!(Json::parse("\"hi\"").unwrap(), Json::String("hi".into()));
    }

    #[test]
    fn parses_nested_structures_in_order() {
        let doc = Json::parse(r#"{"b": [1, 2, {"c": null}], "a": {}, "e": []}"#).unwrap();
        let keys: Vec<&str> = doc.as_object().unwrap().iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a", "e"]);
        let b = doc.get("b").unwrap().as_array().unwrap();
        assert_eq!(b[1].as_usize(), Some(2));
        assert_eq!(b[2].get("c"), Some(&Json::Null));
        assert_eq!(doc.get("a"), Some(&Json::Object(vec![])));
        assert_eq!(doc.get("missing"), None);
    }

    #[test]
    fn resolves_escapes() {
        let s = Json::parse(r#""a\"b\\c\/d\n\t\u0041\u00e9\u0120""#).unwrap();
        assert_eq!(s.as_str(), Some("a\"b\\c/d\n\tA\u{e9}\u{120}"));
    }

    #[test]
    fn joins_surrogate_pairs() {
        // U+1F600 (a grinning face) is D83D DE00 as a surrogate pair.
        let s = Json::parse(r#""\ud83d\ude00""#).unwrap();
        assert_eq!(s.as_str(), Some("\u{1F600}"));
        assert!(Json::parse(r#""\ud83d""#).is_err());
        assert!(Json::parse(r#""\ude00""#).is_err());
    }

    #[test]
    fn passes_raw_utf8_through() {
        let s = Json::parse("\"Ġhello — 世界\"").unwrap();
        assert_eq!(s.as_str(), Some("Ġhello — 世界"));
    }

    #[test]
    fn as_usize_refuses_fractions_and_negatives() {
        assert_eq!(Json::Number(3.0).as_usize(), Some(3));
        assert_eq!(Json::Number(3.5).as_usize(), None);
        assert_eq!(Json::Number(-3.0).as_usize(), None);
    }

    #[test]
    fn rejects_malformed_documents() {
        for bad in ["", "{", "[1,]", "{\"a\" 1}", "tru", "\"abc", "1 2", "{\"a\":1,}", "\"\\x\"", "-"] {
            assert!(Json::parse(bad).is_err(), "should reject {bad:?}");
        }
        let deep = "[".repeat(1000);
        assert!(Json::parse(&deep).is_err());
    }
}
