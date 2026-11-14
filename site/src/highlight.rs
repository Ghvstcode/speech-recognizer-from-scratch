//! A small syntax highlighter for the Rust listings.
//!
//! It is a scanner, not a parser: it walks the characters once and wraps
//! comments, strings, numbers, keywords, types, macros and function names in
//! `<span class="...">`. Everything it emits is HTML-escaped.

const KEYWORDS: &[&str] = &[
    "as", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern", "false", "fn",
    "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref",
    "return", "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe",
    "use", "where", "while",
];

const PRIMITIVES: &[&str] = &[
    "bool", "char", "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "str", "u8", "u16",
    "u32", "u64", "u128", "usize",
];

pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

fn span(out: &mut String, class: &str, text: &str) {
    out.push_str("<span class=\"");
    out.push_str(class);
    out.push_str("\">");
    out.push_str(&escape(text));
    out.push_str("</span>");
}

fn text_of(c: &[char]) -> String {
    c.iter().collect()
}

pub fn rust(code: &str) -> String {
    let c: Vec<char> = code.chars().collect();
    let n = c.len();
    let mut out = String::with_capacity(code.len() * 2);
    let mut i = 0;
    // True right after the keyword `fn`, so the next identifier is a function name.
    let mut after_fn = false;

    while i < n {
        let ch = c[i];

        // Line comment.
        if ch == '/' && i + 1 < n && c[i + 1] == '/' {
            let mut j = i;
            while j < n && c[j] != '\n' {
                j += 1;
            }
            span(&mut out, "c", &text_of(&c[i..j]));
            i = j;
            continue;
        }

        // Block comment.
        if ch == '/' && i + 1 < n && c[i + 1] == '*' {
            let mut j = i + 2;
            while j + 1 < n && !(c[j] == '*' && c[j + 1] == '/') {
                j += 1;
            }
            let end = (j + 2).min(n);
            span(&mut out, "c", &text_of(&c[i..end]));
            i = end;
            continue;
        }

        // Attribute: #[...] or #![...].
        if ch == '#' && i + 1 < n && (c[i + 1] == '[' || (c[i + 1] == '!' && i + 2 < n && c[i + 2] == '[')) {
            let mut j = i;
            let mut depth = 0;
            while j < n {
                if c[j] == '[' {
                    depth += 1;
                } else if c[j] == ']' {
                    depth -= 1;
                    if depth == 0 {
                        j += 1;
                        break;
                    }
                } else if c[j] == '\n' {
                    break;
                }
                j += 1;
            }
            span(&mut out, "a", &text_of(&c[i..j]));
            i = j;
            continue;
        }

        // Raw string: r"..." or r#"..."#, with an optional b in front.
        if (ch == 'r' || (ch == 'b' && i + 1 < n && c[i + 1] == 'r')) && !prev_is_ident(&c, i) {
            let start = if ch == 'b' { i + 2 } else { i + 1 };
            let mut hashes = 0;
            let mut j = start;
            while j < n && c[j] == '#' {
                hashes += 1;
                j += 1;
            }
            if j < n && c[j] == '"' {
                j += 1;
                loop {
                    if j >= n {
                        break;
                    }
                    if c[j] == '"' {
                        let mut k = 0;
                        while k < hashes && j + 1 + k < n && c[j + 1 + k] == '#' {
                            k += 1;
                        }
                        if k == hashes {
                            j += 1 + hashes;
                            break;
                        }
                    }
                    j += 1;
                }
                span(&mut out, "s", &text_of(&c[i..j]));
                i = j;
                continue;
            }
        }

        // String, with an optional b in front.
        if ch == '"' || (ch == 'b' && i + 1 < n && c[i + 1] == '"' && !prev_is_ident(&c, i)) {
            let mut j = if ch == 'b' { i + 2 } else { i + 1 };
            while j < n && c[j] != '"' {
                if c[j] == '\\' {
                    j += 1;
                }
                j += 1;
            }
            let end = (j + 1).min(n);
            span(&mut out, "s", &text_of(&c[i..end]));
            i = end;
            continue;
        }

        // A quote is either a character literal or a lifetime.
        if ch == '\'' || (ch == 'b' && i + 1 < n && c[i + 1] == '\'' && !prev_is_ident(&c, i)) {
            let q = if ch == 'b' { i + 1 } else { i };
            if q + 1 < n && c[q + 1] == '\\' {
                // '\n', '\'', '\u{1F600}'
                let mut j = q + 3;
                while j < n && c[j] != '\'' && c[j] != '\n' {
                    j += 1;
                }
                let end = (j + 1).min(n);
                span(&mut out, "s", &text_of(&c[i..end]));
                i = end;
                continue;
            }
            if q + 2 < n && c[q + 2] == '\'' {
                span(&mut out, "s", &text_of(&c[i..q + 3]));
                i = q + 3;
                continue;
            }
            if ch == '\'' {
                let mut j = i + 1;
                while j < n && (c[j].is_alphanumeric() || c[j] == '_') {
                    j += 1;
                }
                if j > i + 1 {
                    span(&mut out, "l", &text_of(&c[i..j]));
                    i = j;
                    continue;
                }
            }
        }

        // Number.
        if ch.is_ascii_digit() && !prev_is_ident(&c, i) {
            let mut j = i;
            while j < n {
                let d = c[j];
                let exponent_sign = (d == '-' || d == '+')
                    && j > i
                    && (c[j - 1] == 'e' || c[j - 1] == 'E')
                    && !text_of(&c[i..j]).starts_with("0x")
                    && j + 1 < n
                    && c[j + 1].is_ascii_digit();
                let decimal_point = d == '.' && j + 1 < n && c[j + 1].is_ascii_digit();
                if d.is_alphanumeric() || d == '_' || decimal_point || exponent_sign {
                    j += 1;
                } else {
                    break;
                }
            }
            span(&mut out, "n", &text_of(&c[i..j]));
            i = j;
            continue;
        }

        // Identifier, keyword, type, macro or function name.
        if ch.is_alphabetic() || ch == '_' {
            let mut j = i;
            while j < n && (c[j].is_alphanumeric() || c[j] == '_') {
                j += 1;
            }
            let word = text_of(&c[i..j]);
            let is_macro = j < n && c[j] == '!' && !(j + 1 < n && c[j + 1] == '=');
            if is_macro {
                span(&mut out, "m", &format!("{word}!"));
                i = j + 1;
                after_fn = false;
                continue;
            }
            if KEYWORDS.contains(&word.as_str()) {
                span(&mut out, "k", &word);
                after_fn = word == "fn";
            } else if after_fn {
                span(&mut out, "f", &word);
                after_fn = false;
            } else if PRIMITIVES.contains(&word.as_str())
                || (word.chars().next().is_some_and(char::is_uppercase)
                    && word.chars().any(char::is_lowercase))
            {
                span(&mut out, "t", &word);
            } else {
                out.push_str(&escape(&word));
            }
            i = j;
            continue;
        }

        if !ch.is_whitespace() {
            after_fn = false;
        }
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
        i += 1;
    }
    out
}

fn prev_is_ident(c: &[char], i: usize) -> bool {
    i > 0 && (c[i - 1].is_alphanumeric() || c[i - 1] == '_')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_generics() {
        let html = rust("let v: Vec<f32> = Vec::new();");
        assert!(html.contains("Vec</span>&lt;<span class=\"t\">f32</span>&gt;"), "{html}");
    }

    #[test]
    fn comments_strings_and_lifetimes() {
        let html = rust("fn name<'a>(s: &'a str) -> char { 'x' } // done \"not a string\"");
        assert!(html.contains("<span class=\"f\">name</span>"), "{html}");
        assert!(html.contains("<span class=\"l\">'a</span>"), "{html}");
        assert!(html.contains("<span class=\"s\">'x'</span>"), "{html}");
        assert!(html.contains("<span class=\"c\">// done &quot;not a string&quot;</span>"), "{html}");
    }

    #[test]
    fn numbers_and_macros() {
        let html = rust("assert_eq!(x, 1.5e-3); let r = 0..n; let y = 1_000u32;");
        assert!(html.contains("<span class=\"m\">assert_eq!</span>"), "{html}");
        assert!(html.contains("<span class=\"n\">1.5e-3</span>"), "{html}");
        assert!(html.contains("<span class=\"n\">0</span>..n"), "{html}");
        assert!(html.contains("<span class=\"n\">1_000u32</span>"), "{html}");
    }

    #[test]
    fn not_equal_is_not_a_macro() {
        let html = rust("if a != b {}");
        assert!(!html.contains("class=\"m\""), "{html}");
    }
}
