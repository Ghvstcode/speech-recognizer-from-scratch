//! Markdown to HTML, for the subset of Markdown the chapters use.
//!
//! Blocks: headings, paragraphs, fenced code, lists (nested, with code and
//! paragraphs inside items), tables, block quotes, standalone images with an
//! italic caption underneath, and the book's two conventions: an "Aside"
//! paragraph and a "Play with it" quote that links an interactive page.
//!
//! Inline: code spans, strong, emphasis, links, images, `<br>`, bare URLs.
//! Any other HTML in the text is escaped, so `Vec<f32>` in prose is safe.

use crate::highlight::{self, escape};
use std::collections::HashSet;

/// A `###` section of the page, collected for the table of contents.
pub struct Heading {
    pub id: String,
    pub html: String,
}

pub struct Markdown<'a> {
    /// Turns a link target as written in the Markdown file into the target
    /// for the generated page. `None` means "there is nowhere to link to":
    /// the text is kept and the link is dropped.
    pub resolve: &'a dyn Fn(&str) -> Option<String>,
    /// The sections found so far.
    pub headings: Vec<Heading>,
    used_ids: HashSet<String>,
}

struct Marker {
    indent: usize,
    ordered: bool,
    /// Column where the item's text starts.
    content: usize,
    /// The number an ordered item was written with.
    number: usize,
}

fn leading_spaces(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

fn marker(line: &str) -> Option<Marker> {
    let indent = leading_spaces(line);
    let rest = &line[indent..];
    if rest.starts_with("- ") || rest.starts_with("* ") || rest.starts_with("+ ") {
        return Some(Marker { indent, ordered: false, content: indent + 2, number: 0 });
    }
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    if digits > 0 && digits < 4 && rest[digits..].starts_with(". ") {
        let number = rest[..digits].parse().unwrap_or(1);
        return Some(Marker { indent, ordered: true, content: indent + digits + 2, number });
    }
    None
}

fn is_fence(trimmed: &str) -> bool {
    trimmed.starts_with("```")
}

fn heading(trimmed: &str) -> Option<&str> {
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    if (1..=6).contains(&hashes) && trimmed[hashes..].starts_with(' ') {
        Some(trimmed[hashes..].trim())
    } else {
        None
    }
}

fn is_table_separator(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|') && t.contains("--") && t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

fn table_cells(line: &str) -> Vec<String> {
    let t = line.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut chars = t.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' && chars.peek() == Some(&'|') {
            cell.push('|');
            chars.next();
        } else if ch == '|' {
            cells.push(cell.trim().to_string());
            cell = String::new();
        } else {
            cell.push(ch);
        }
    }
    cells.push(cell.trim().to_string());
    cells
}

/// `![alt](src)` and nothing else on the line.
fn lone_image(trimmed: &str) -> Option<(String, String)> {
    if !trimmed.starts_with("![") || !trimmed.ends_with(')') {
        return None;
    }
    let c: Vec<char> = trimmed.chars().collect();
    let (text, url, end) = link_at(&c, 1)?;
    if end != c.len() {
        return None;
    }
    Some((text.iter().collect(), url))
}

/// A whole line in single asterisks: the caption under a figure.
fn caption(trimmed: &str) -> Option<&str> {
    if trimmed.len() > 2 && trimmed.starts_with('*') && !trimmed.starts_with("**") && trimmed.ends_with('*') {
        Some(&trimmed[1..trimmed.len() - 1])
    } else {
        None
    }
}

/// Parses `[text](url)` with the `[` at `c[open]`.
/// Returns the text, the URL and the index just past the `)`.
fn link_at(c: &[char], open: usize) -> Option<(&[char], String, usize)> {
    if c.get(open) != Some(&'[') {
        return None;
    }
    let mut depth = 0;
    let mut j = open;
    let close = loop {
        if j >= c.len() {
            return None;
        }
        match c[j] {
            '\\' => j += 1,
            '`' => j = skip_code_span(c, j).unwrap_or(j + 1) - 1,
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    break j;
                }
            }
            _ => {}
        }
        j += 1;
    };
    if c.get(close + 1) != Some(&'(') {
        return None;
    }
    let mut depth = 0;
    let mut j = close + 1;
    let end = loop {
        if j >= c.len() {
            return None;
        }
        match c[j] {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    break j;
                }
            }
            _ => {}
        }
        j += 1;
    };
    let url: String = c[close + 2..end].iter().collect();
    // Drop an optional title: [text](url "title").
    let url = url.trim().split(" \"").next().unwrap_or("").to_string();
    Some((&c[open + 1..close], url, end + 1))
}

/// With a backtick at `c[i]`, returns the index just past the closing run.
fn skip_code_span(c: &[char], i: usize) -> Option<usize> {
    let mut run = 0;
    while i + run < c.len() && c[i + run] == '`' {
        run += 1;
    }
    let mut j = i + run;
    while j < c.len() {
        if c[j] == '`' {
            let mut r = 0;
            while j + r < c.len() && c[j + r] == '`' {
                r += 1;
            }
            if r == run {
                return Some(j + r);
            }
            j += r;
        } else {
            j += 1;
        }
    }
    None
}

/// Finds the `*` or `**` that closes emphasis opened just before `from`.
fn find_closer(c: &[char], from: usize, stars: usize) -> Option<usize> {
    let mut j = from;
    while j < c.len() {
        match c[j] {
            '\\' => j += 2,
            '`' => j = skip_code_span(c, j).unwrap_or(j + 1),
            '*' => {
                let mut run = 0;
                while j + run < c.len() && c[j + run] == '*' {
                    run += 1;
                }
                let after_text = j > from && !c[j - 1].is_whitespace();
                if run == stars && after_text {
                    return Some(j);
                }
                if stars == 1 && run == 2 {
                    // Strong inside emphasis: jump over it.
                    j = find_closer(c, j + 2, 2).map_or(j + 2, |k| k + 2);
                } else {
                    j += run;
                }
            }
            _ => j += 1,
        }
    }
    None
}

fn plain_text(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    out.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"")
}

impl<'a> Markdown<'a> {
    pub fn new(resolve: &'a dyn Fn(&str) -> Option<String>) -> Self {
        Markdown { resolve, headings: Vec::new(), used_ids: HashSet::new() }
    }

    pub fn render(&mut self, source: &str) -> String {
        let lines: Vec<String> = source.lines().map(str::to_string).collect();
        self.blocks(&lines)
    }

    /// GitHub's rule: lowercase, keep letters, digits, `-` and `_`, turn
    /// spaces into `-`, drop the rest. A repeated heading gets `-1`, `-2`.
    fn slug(&mut self, text: &str) -> String {
        let mut base = String::new();
        for ch in text.chars() {
            if ch.is_alphanumeric() || ch == '-' || ch == '_' {
                base.extend(ch.to_lowercase());
            } else if ch == ' ' {
                base.push('-');
            }
        }
        let mut id = base.clone();
        let mut n = 1;
        while !self.used_ids.insert(id.clone()) {
            id = format!("{base}-{n}");
            n += 1;
        }
        id
    }

    fn blocks(&mut self, lines: &[String]) -> String {
        let mut out = String::new();
        let mut i = 0;
        while i < lines.len() {
            let line = &lines[i];
            let t = line.trim();
            if t.is_empty() {
                i += 1;
                continue;
            }

            // Fenced code.
            if is_fence(t) {
                let indent = leading_spaces(line);
                let ticks = t.chars().take_while(|&c| c == '`').count();
                let lang = t[ticks..].trim();
                let mut code = Vec::new();
                i += 1;
                while i < lines.len() {
                    let l = lines[i].trim();
                    if l.len() >= ticks && l.chars().all(|c| c == '`') {
                        break;
                    }
                    let strip = leading_spaces(&lines[i]).min(indent);
                    code.push(&lines[i][strip..]);
                    i += 1;
                }
                i += 1;
                out.push_str(&self.code_block(lang, &code));
                continue;
            }

            // Heading. The page title is taken out before we get here, so
            // every heading in the body is a section.
            if let Some(text) = heading(t) {
                let html = self.inline(text);
                let id = self.slug(&plain_text(&html));
                out.push_str(&format!(
                    "<h2 id=\"{id}\"><a class=\"anchor\" href=\"#{id}\" aria-hidden=\"true\" tabindex=\"-1\">#</a>{html}</h2>\n"
                ));
                self.headings.push(Heading { id, html });
                i += 1;
                continue;
            }

            // Block quote.
            if t.starts_with('>') {
                let mut inner = Vec::new();
                while i < lines.len() && lines[i].trim_start().starts_with('>') {
                    let l = lines[i].trim_start();
                    let l = l.strip_prefix('>').unwrap_or(l);
                    inner.push(l.strip_prefix(' ').unwrap_or(l).to_string());
                    i += 1;
                }
                if let Some(html) = self.play(&inner) {
                    out.push_str(&html);
                } else {
                    out.push_str("<blockquote>\n");
                    out.push_str(&self.blocks(&inner));
                    out.push_str("</blockquote>\n");
                }
                continue;
            }

            // Table.
            if t.starts_with('|') && i + 1 < lines.len() && is_table_separator(&lines[i + 1]) {
                let head = table_cells(t);
                let align: Vec<&str> = table_cells(&lines[i + 1])
                    .iter()
                    .map(|c| match (c.starts_with(':'), c.ends_with(':')) {
                        (true, true) => " class=\"center\"",
                        (false, true) => " class=\"right\"",
                        _ => "",
                    })
                    .collect();
                let a = |k: usize| align.get(k).copied().unwrap_or("");
                out.push_str("<div class=\"table-wrap\"><table>\n<thead><tr>");
                for (k, cell) in head.iter().enumerate() {
                    out.push_str(&format!("<th{}>{}</th>", a(k), self.inline(cell)));
                }
                out.push_str("</tr></thead>\n<tbody>\n");
                i += 2;
                while i < lines.len() && lines[i].trim_start().starts_with('|') {
                    out.push_str("<tr>");
                    for (k, cell) in table_cells(&lines[i]).iter().enumerate() {
                        out.push_str(&format!("<td{}>{}</td>", a(k), self.inline(cell)));
                    }
                    out.push_str("</tr>\n");
                    i += 1;
                }
                out.push_str("</tbody>\n</table></div>\n");
                continue;
            }

            // List.
            if let Some(first) = marker(line).filter(|m| m.indent <= 3) {
                i = self.list(lines, i, &first, &mut out);
                continue;
            }

            // A figure: an image alone on its line, and an italic caption under it.
            if let Some((alt, src)) = lone_image(t) {
                let mut j = i + 1;
                while j < lines.len() && lines[j].trim().is_empty() {
                    j += 1;
                }
                let cap = lines.get(j).and_then(|l| caption(l.trim()));
                let src = (self.resolve)(&src).unwrap_or(src);
                out.push_str(&format!(
                    "<figure class=\"figure\"><img src=\"{}\" alt=\"{}\" loading=\"lazy\">",
                    escape(&src),
                    escape(&alt)
                ));
                if let Some(cap) = cap {
                    out.push_str(&format!("<figcaption>{}</figcaption>", self.inline(cap)));
                    i = j + 1;
                } else {
                    i += 1;
                }
                out.push_str("</figure>\n");
                continue;
            }

            if t == "---" || t == "***" {
                out.push_str("<hr>\n");
                i += 1;
                continue;
            }

            // Paragraph: everything up to a blank line or the start of another block.
            let mut para = vec![t];
            i += 1;
            while i < lines.len() {
                let l = lines[i].trim();
                let starts_block = l.is_empty()
                    || is_fence(l)
                    || heading(l).is_some()
                    || l.starts_with('>')
                    || marker(&lines[i]).is_some_and(|m| !m.ordered || m.number == 1);
                if starts_block {
                    break;
                }
                para.push(l);
                i += 1;
            }
            let text = para.join("\n");

            // The line of previous/next links at the foot of a chapter: the
            // page template draws its own.
            if text.starts_with("[←") {
                continue;
            }
            if let Some(rest) = text.strip_prefix("**Aside**") {
                let rest = rest.trim_start();
                let mut rest = rest.strip_prefix("<br>").unwrap_or(rest).trim_start().to_string();
                if rest.is_empty() {
                    // The label is on a line of its own, with the text as
                    // the next paragraph.
                    while i < lines.len() && lines[i].trim().is_empty() {
                        i += 1;
                    }
                    let mut next = Vec::new();
                    while i < lines.len() && !lines[i].trim().is_empty() && heading(lines[i].trim()).is_none() {
                        next.push(lines[i].trim());
                        i += 1;
                    }
                    rest = next.join("\n");
                }
                let rest = rest.as_str();
                out.push_str(&format!(
                    "<aside class=\"aside\"><span class=\"overline\">Aside</span><p>{}</p></aside>\n",
                    self.inline(rest)
                ));
                continue;
            }
            out.push_str(&format!("<p>{}</p>\n", self.inline(&text)));
        }
        out
    }

    /// Renders the list that starts at `lines[start]`; returns the index of
    /// the first line after it.
    fn list(&mut self, lines: &[String], start: usize, first: &Marker, out: &mut String) -> usize {
        let base = first.indent;
        let mut items: Vec<Vec<String>> = Vec::new();
        let mut content = first.content;
        let mut loose = false;
        let mut j = start;
        while j < lines.len() {
            let line = &lines[j];
            if line.trim().is_empty() {
                let mut k = j;
                while k < lines.len() && lines[k].trim().is_empty() {
                    k += 1;
                }
                let Some(next) = lines.get(k) else { break };
                let same_list = marker(next).is_some_and(|m| m.indent <= base + 1 && m.ordered == first.ordered);
                if same_list {
                    loose = true;
                } else if leading_spaces(next) >= base + 2 {
                    // A second paragraph, a code block or a nested list inside the item.
                    if let Some(item) = items.last_mut() {
                        item.push(String::new());
                    }
                } else {
                    break;
                }
                j = k;
                continue;
            }
            match marker(line) {
                Some(m) if m.indent <= base + 1 => {
                    if m.ordered != first.ordered {
                        break;
                    }
                    content = m.content;
                    items.push(vec![line[m.content..].to_string()]);
                }
                _ => {
                    let indent = leading_spaces(line);
                    if indent < base + 2 {
                        break;
                    }
                    if let Some(item) = items.last_mut() {
                        item.push(line[indent.min(content)..].to_string());
                    }
                }
            }
            j += 1;
        }

        let tag = if first.ordered { "ol" } else { "ul" };
        if first.ordered && first.number != 1 {
            out.push_str(&format!("<ol start=\"{}\">\n", first.number));
        } else {
            out.push_str(&format!("<{tag}>\n"));
        }
        for item in &items {
            let mut html = self.blocks(item);
            if !loose && html.starts_with("<p>") {
                // A tight list: the item's text sits directly in the <li>.
                html = html.replacen("<p>", "", 1).replacen("</p>", "", 1);
            }
            out.push_str(&format!("<li>{}</li>\n", html.trim_end()));
        }
        out.push_str(&format!("</{tag}>\n"));
        j
    }

    /// `> **Play with it:** [title](../interactive/page.html). What to try.`
    /// becomes the interactive page itself, embedded where the reader is.
    fn play(&mut self, quote: &[String]) -> Option<String> {
        let text = quote.join(" ");
        let rest = text.trim().strip_prefix("**Play with it:**")?.trim_start();
        let c: Vec<char> = rest.chars().collect();
        let (title, url, end) = link_at(&c, 0)?;
        if !url.contains("interactive/") {
            return None;
        }
        let href = (self.resolve)(&url)?;
        let title: String = title.iter().collect();
        let title_html = self.inline(&title);
        let note: String = c[end..].iter().collect();
        let note = note.trim_start_matches(['.', ',', ' ']).trim();
        let mut html = format!(
            "<figure class=\"play\">\
             <div class=\"play-head\"><span class=\"overline\">Play with it</span>\
             <span class=\"play-title\">{title_html}</span>\
             <a class=\"play-open\" href=\"{href}\">Open full page<span aria-hidden=\"true\"> ↗</span></a></div>\
             <iframe src=\"{href}?embed=1\" title=\"Interactive: {}\" loading=\"lazy\"></iframe>",
            escape(&plain_text(&title_html)),
            href = escape(&href),
        );
        if !note.is_empty() {
            html.push_str(&format!("<figcaption>{}</figcaption>", self.inline(note)));
        }
        html.push_str("</figure>\n");
        Some(html)
    }

    fn code_block(&mut self, lang: &str, lines: &[&str]) -> String {
        let mut lines = lines;
        let mut label = String::new();
        // Listings start with a comment naming their file: `// src/dsp/fft.rs`.
        // That line becomes the header of the block.
        if let Some(first) = lines.first() {
            if let Some(path) = first.strip_prefix("// ") {
                let file = path.split_whitespace().next().unwrap_or("");
                let looks_like_path = (file.starts_with("src/") || file.starts_with("examples/") || file.starts_with("tests/"))
                    && file.ends_with(".rs")
                    || file == "Cargo.toml";
                if looks_like_path {
                    let text = escape(path);
                    label = match (self.resolve)(&format!("/code/{file}")) {
                        Some(url) => {
                            let tail = escape(&path[file.len()..]);
                            format!("<a href=\"{}\">{}</a>{tail}", escape(&url), escape(file))
                        }
                        None => text,
                    };
                    lines = &lines[1..];
                    while lines.first().is_some_and(|l| l.trim().is_empty()) {
                        lines = &lines[1..];
                    }
                }
            }
        }
        let code = lines.join("\n");
        let body = if lang == "rust" { highlight::rust(&code) } else { escape(&code) };
        let kind = if lang.is_empty() { "text" } else { lang };
        let mut html = format!("<figure class=\"code code-{}\">", escape(kind));
        if !label.is_empty() {
            html.push_str(&format!("<figcaption>{label}</figcaption>"));
        }
        html.push_str(&format!("<pre><code>{body}\n</code></pre></figure>\n"));
        html
    }

    pub fn inline(&mut self, text: &str) -> String {
        let c: Vec<char> = text.chars().collect();
        self.spans(&c)
    }

    fn spans(&mut self, c: &[char]) -> String {
        let mut out = String::new();
        let mut i = 0;
        let starts = |i: usize, s: &str| {
            let s: Vec<char> = s.chars().collect();
            c.len() >= i + s.len() && c[i..i + s.len()] == s[..]
        };
        while i < c.len() {
            let ch = c[i];
            match ch {
                '\\' if i + 1 < c.len() && c[i + 1].is_ascii_punctuation() => {
                    out.push_str(&escape(&c[i + 1].to_string()));
                    i += 2;
                }
                '`' => match skip_code_span(c, i) {
                    Some(end) => {
                        let run = c[i..].iter().take_while(|&&x| x == '`').count();
                        let inner: String = c[i + run..end - run].iter().collect();
                        let inner = if inner.starts_with(' ') && inner.ends_with(' ') && inner.trim() != "" {
                            inner[1..inner.len() - 1].to_string()
                        } else {
                            inner
                        };
                        out.push_str(&format!("<code>{}</code>", escape(&inner)));
                        i = end;
                    }
                    None => {
                        out.push('`');
                        i += 1;
                    }
                },
                '!' if starts(i + 1, "[") => match link_at(c, i + 1) {
                    Some((alt, url, end)) => {
                        let alt: String = alt.iter().collect();
                        let src = (self.resolve)(&url).unwrap_or(url);
                        out.push_str(&format!("<img src=\"{}\" alt=\"{}\">", escape(&src), escape(&alt)));
                        i = end;
                    }
                    None => {
                        out.push('!');
                        i += 1;
                    }
                },
                '[' => match link_at(c, i) {
                    Some((label, url, end)) => {
                        let label = self.spans(label);
                        match (self.resolve)(&url) {
                            Some(href) => out.push_str(&format!("<a href=\"{}\">{label}</a>", escape(&href))),
                            None => out.push_str(&label),
                        }
                        i = end;
                    }
                    None => {
                        out.push('[');
                        i += 1;
                    }
                },
                '*' => {
                    let stars = if starts(i, "**") { 2 } else { 1 };
                    let opens = c.get(i + stars).is_some_and(|x| !x.is_whitespace());
                    match find_closer(c, i + stars, stars).filter(|_| opens) {
                        Some(close) => {
                            let tag = if stars == 2 { "strong" } else { "em" };
                            let inner = self.spans(&c[i + stars..close]);
                            out.push_str(&format!("<{tag}>{inner}</{tag}>"));
                            i = close + stars;
                        }
                        None => {
                            out.push('*');
                            i += 1;
                        }
                    }
                }
                '<' if starts(i, "<br>") => {
                    out.push_str("<br>");
                    i += 4;
                }
                'h' if (starts(i, "http://") || starts(i, "https://"))
                    && (i == 0 || !(c[i - 1].is_alphanumeric() || c[i - 1] == '/')) =>
                {
                    let mut end = i;
                    while end < c.len() && !c[end].is_whitespace() && !matches!(c[end], '<' | '>' | ')' | '"') {
                        end += 1;
                    }
                    while end > i && matches!(c[end - 1], '.' | ',' | ';' | ':' | '!' | '?') {
                        end -= 1;
                    }
                    let url: String = c[i..end].iter().collect();
                    out.push_str(&format!("<a href=\"{0}\">{0}</a>", escape(&url)));
                    i = end;
                }
                '\n' => {
                    out.push('\n');
                    i += 1;
                }
                _ => {
                    out.push_str(&escape(&ch.to_string()));
                    i += 1;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render(source: &str) -> String {
        let resolve = |url: &str| -> Option<String> {
            if url.starts_with("/code/") {
                None
            } else {
                Some(url.replace(".md", ".html"))
            }
        };
        Markdown::new(&resolve).render(source)
    }

    #[test]
    fn prose_html_is_escaped_but_br_survives() {
        let html = render("A `Vec<f32>` and a bare Vec<f32> and 1 < 2 & so on.");
        assert_eq!(html, "<p>A <code>Vec&lt;f32&gt;</code> and a bare Vec&lt;f32&gt; and 1 &lt; 2 &amp; so on.</p>\n");
    }

    #[test]
    fn emphasis_strong_and_links() {
        let html = render("We *really* do **write [`dtw.rs`](../chapters/05.md)** and 2 * 3 * 4 stays.");
        assert_eq!(
            html,
            "<p>We <em>really</em> do <strong>write <a href=\"../chapters/05.html\"><code>dtw.rs</code></a></strong> and 2 * 3 * 4 stays.</p>\n"
        );
    }

    #[test]
    fn nested_list_with_code() {
        let html = render("1. One\n2. Two\n\n   ```rust\n   let x = 1;\n   ```\n\n   - inner\n3. Three\n\nAfter.");
        assert!(html.starts_with("<ol>\n<li><p>One</p></li>") || html.starts_with("<ol>\n<li>One</li>"), "{html}");
        assert!(html.contains("<figure class=\"code code-rust\"><pre><code><span class=\"k\">let</span> x"), "{html}");
        assert!(html.contains("<ul>\n<li>inner</li>\n</ul>"), "{html}");
        assert!(html.contains("<li>Three</li>\n</ol>\n<p>After.</p>"), "{html}");
    }

    #[test]
    fn table_and_figure() {
        let html = render("| A | B |\n|---|--:|\n| `x\\|y` | 2 |\n\n![alt text](../figures/a.svg)\n\n*The caption.*\n\nNext.");
        assert!(html.contains("<th>A</th><th class=\"right\">B</th>"), "{html}");
        assert!(html.contains("<td><code>x|y</code></td><td class=\"right\">2</td>"), "{html}");
        assert!(html.contains("<img src=\"../figures/a.svg\" alt=\"alt text\" loading=\"lazy\"><figcaption>The caption.</figcaption>"), "{html}");
        assert!(html.ends_with("<p>Next.</p>\n"), "{html}");
    }

    #[test]
    fn play_quote_becomes_an_embed() {
        let html = render("> **Play with it:** [the DTW grid](../interactive/dtw.html). Drag a point.");
        assert!(html.contains("<iframe src=\"../interactive/dtw.html?embed=1\""), "{html}");
        assert!(html.contains("<figcaption>Drag a point.</figcaption>"), "{html}");
    }

    #[test]
    fn aside_listing_label_and_nav_line() {
        let html = render("**Aside** <br>\nA trap.\n\n```rust\n// src/dtw.rs\n\npub fn dtw() {}\n```\n\n[← Contents](../README.md) · [Next: X →](02.md)\n\n### Loose Ends\n\n### Loose Ends");
        assert!(html.contains("<aside class=\"aside\"><span class=\"overline\">Aside</span><p>A trap.</p></aside>"), "{html}");
        assert!(html.contains("<figcaption>src/dtw.rs</figcaption><pre><code><span class=\"k\">pub</span>"), "{html}");
        assert!(!html.contains("Contents"), "{html}");
        assert!(html.contains("id=\"loose-ends\"") && html.contains("id=\"loose-ends-1\""), "{html}");
    }
}
