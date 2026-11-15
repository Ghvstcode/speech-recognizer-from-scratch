//! Builds the book's website.
//!
//!     cargo run --release --manifest-path site/Cargo.toml
//!
//! Reads `README.md` and `chapters/*.md`, writes `_site/`. The Markdown
//! files stay the only copy of the text: this program turns each one into a
//! page, puts the interactive pages inline where a chapter links to them,
//! and copies the figures, the interactive pages and the sample recordings.
//!
//! Three optional environment variables:
//!
//! * `SITE_REPO_URL`: the repository's address, for example
//!   `https://github.com/someone/speech`. Links to source files point there.
//!   Without it, those links are left as plain text.
//! * `SITE_REPO_BRANCH`: the branch those links use. Defaults to `main`.
//! * `SITE_ANALYTICS_TOKEN`: a Cloudflare Web Analytics token. With it, every
//!   page loads Cloudflare's visit counter (no cookies, nothing stored in the
//!   reader's browser). Without it, the pages load nothing from elsewhere.

mod highlight;
mod markdown;

use highlight::escape;
use markdown::Markdown;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

const BOOK: &str = "Building a Speech Recognizer from Scratch";

/// The four parts of the book and the chapters each one covers.
const PARTS: &[(&str, &str, std::ops::RangeInclusive<usize>)] = &[
    ("I", "Hearing", 1..=4),
    ("II", "A Recognizer With No Learning", 5..=5),
    ("III", "Learning", 6..=11),
    ("IV", "Standing On Shoulders", 12..=14),
];

struct Chapter {
    number: usize,
    /// File name without the extension: `05-dynamic-time-warping`.
    stem: String,
    title: String,
    /// The Markdown after the title line.
    body: String,
    minutes: usize,
}

struct Site {
    repo: Option<String>,
    branch: String,
    /// Cloudflare Web Analytics token; `None` leaves the counter out.
    analytics: Option<String>,
}

const BEACON: &str = "https://static.cloudflareinsights.com/beacon.min.js";

/// A token is letters and digits. Anything else is refused, since the token
/// is written into every page.
fn analytics_token(value: Option<String>) -> Option<String> {
    let token = value?.trim().to_string();
    let plain = !token.is_empty() && token.chars().all(|c| c.is_ascii_alphanumeric());
    plain.then_some(token)
}

impl Site {
    /// The visit counter for a chapter or the front page.
    fn counter(&self) -> String {
        match &self.analytics {
            Some(token) => {
                format!("<script defer src=\"{BEACON}\" data-cf-beacon='{{\"token\": \"{token}\"}}'></script>\n")
            }
            None => String::new(),
        }
    }

    /// The visit counter for an interactive page. A page shown inside a
    /// chapter is not counted: the chapter already is.
    fn counter_unless_embedded(&self) -> String {
        match &self.analytics {
            Some(token) => format!(
                r#"<script>
if (!/[?&]embed\b/.test(location.search) || window.parent === window) {{
  var counter = document.createElement('script');
  counter.defer = true;
  counter.src = '{BEACON}';
  counter.setAttribute('data-cf-beacon', '{{"token": "{token}"}}');
  document.body.appendChild(counter);
}}
</script>
"#
            ),
            None => String::new(),
        }
    }

    /// Maps a link written in a Markdown file to a link that works on the
    /// generated page. `from_dir` is the directory of the Markdown file
    /// relative to the repository root (`""` or `"chapters"`); the page is
    /// written to the same directory under `_site/`.
    fn resolve(&self, url: &str, from_dir: &str) -> Option<String> {
        if url.starts_with('#') || url.contains("://") || url.starts_with("mailto:") {
            return Some(url.to_string());
        }
        let (path, fragment) = match url.split_once('#') {
            Some((p, f)) => (p, format!("#{f}")),
            None => (url, String::new()),
        };

        // Path relative to the repository root. A leading slash means it
        // already is one.
        let mut parts: Vec<&str> = if path.starts_with('/') {
            Vec::new()
        } else {
            from_dir.split('/').filter(|p| !p.is_empty()).collect()
        };
        for piece in path.split('/') {
            match piece {
                "" | "." => {}
                ".." => {
                    parts.pop();
                }
                p => parts.push(p),
            }
        }
        let repo_path = parts.join("/");
        let up = if from_dir.is_empty() { "" } else { "../" };

        let on_site = if repo_path == "README.md" || repo_path.is_empty() {
            Some("index.html".to_string())
        } else if repo_path == "chapters" {
            return Some(format!("{up}index.html#how-to-read-this-book"));
        } else if repo_path == "interactive" {
            Some("interactive/index.html".to_string())
        } else if repo_path.starts_with("chapters/") && repo_path.ends_with(".md") {
            Some(repo_path.replace(".md", ".html"))
        } else if repo_path.starts_with("figures/")
            || (repo_path.starts_with("interactive/") && !repo_path.ends_with(".md"))
            || (repo_path.starts_with("samples/") && repo_path.ends_with(".wav"))
        {
            Some(repo_path.clone())
        } else {
            None
        };
        if let Some(target) = on_site {
            return Some(format!("{up}{target}{fragment}"));
        }

        // Everything else (source files, models, notes) lives in the repository.
        let repo = self.repo.as_ref()?;
        let is_file = parts.last().is_some_and(|name| name.contains('.'));
        let kind = if is_file { "blob" } else { "tree" };
        Some(format!("{repo}/{kind}/{}/{repo_path}{fragment}", self.branch))
    }
}

fn main() -> io::Result<()> {
    let root = match std::env::args().nth(1) {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("site/ has a parent").to_path_buf(),
    };
    let out = root.join("_site");
    let site = Site {
        repo: std::env::var("SITE_REPO_URL").ok().map(|u| u.trim_end_matches('/').to_string()).filter(|u| !u.is_empty()),
        branch: std::env::var("SITE_REPO_BRANCH").unwrap_or_else(|_| "main".to_string()),
        analytics: analytics_token(std::env::var("SITE_ANALYTICS_TOKEN").ok()),
    };

    if out.exists() {
        fs::remove_dir_all(&out)?;
    }
    fs::create_dir_all(out.join("chapters"))?;

    let chapters = read_chapters(&root.join("chapters"))?;

    // The chapters.
    for (index, chapter) in chapters.iter().enumerate() {
        let resolve = |url: &str| {
            // A listing's label links to its file, unless the file is one the
            // chapter has the reader make up (a scratch example).
            if url.strip_prefix('/').is_some_and(|path| !root.join(path).exists()) {
                return None;
            }
            site.resolve(url, "chapters")
        };
        let mut md = Markdown::new(&resolve);
        let content = md.render(&chapter.body);
        let part = PARTS.iter().find(|p| p.2.contains(&chapter.number)).expect("every chapter is in a part");

        let mut article = String::new();
        article.push_str(&format!(
            "<header class=\"chapter-head\"><p class=\"overline\">Part {} · Chapter {} · {} min read</p><h1>{}</h1></header>\n",
            part.0,
            chapter.number,
            chapter.minutes,
            escape(&chapter.title)
        ));
        article.push_str(&content);

        let mut pager = String::from("<nav class=\"pager\" aria-label=\"Chapters\">");
        match index.checked_sub(1).map(|i| &chapters[i]) {
            Some(prev) => pager.push_str(&pager_link("prev", "Previous", &format!("{}.html", prev.stem), prev)),
            None => pager.push_str("<a class=\"pager-link prev\" href=\"../index.html\"><span class=\"overline\">← Back</span><span class=\"pager-title\">Contents</span></a>"),
        }
        match chapters.get(index + 1) {
            Some(next) => pager.push_str(&pager_link("next", "Next", &format!("{}.html", next.stem), next)),
            None => pager.push_str("<a class=\"pager-link next\" href=\"../index.html\"><span class=\"overline\">The end →</span><span class=\"pager-title\">Back to the contents</span></a>"),
        }
        pager.push_str("</nav>\n");

        let page = Page {
            title: format!("{} · {BOOK}", chapter.title),
            description: format!("Chapter {} of {BOOK}: {}.", chapter.number, chapter.title),
            up: "../",
            body_class: "chapter",
            sidebar: sidebar(&chapters, Some((chapter.number, &md.headings)), "../"),
            article,
            after: pager,
        };
        fs::write(out.join("chapters").join(format!("{}.html", chapter.stem)), page.html(&site))?;
    }

    // The front page, from README.md.
    {
        let readme = fs::read_to_string(root.join("README.md"))?;
        let (title, body) = split_title(&readme);
        // The line under the title, in italics, is the tagline.
        let (tagline, body) = match body.trim_start().split_once('\n') {
            Some((first, rest)) if first.starts_with('*') && first.ends_with('*') => {
                (first.trim_matches('*').to_string(), rest.to_string())
            }
            _ => (String::new(), body),
        };
        let resolve = |url: &str| site.resolve(url, "");
        let mut md = Markdown::new(&resolve);
        let content = md.render(&body);
        let first = &chapters[0];
        let article = format!(
            "<header class=\"hero\"><p class=\"overline\">A book · {} chapters · {} dependencies</p>\
             <h1>{}</h1><p class=\"tagline\">{}</p>\
             <p class=\"hero-actions\"><a class=\"button primary\" href=\"chapters/{}.html\">Start reading<span aria-hidden=\"true\"> →</span></a>\
             <a class=\"button\" href=\"interactive/index.html\">Play with the interactives</a></p></header>\n{content}",
            chapters.len(),
            "zero",
            escape(&sentence_case_title(&title)),
            escape(&tagline),
            first.stem,
        );
        let page = Page {
            title: BOOK.to_string(),
            description: format!("{BOOK}: {tagline} A book that builds a speech recognizer with no dependencies, one piece at a time."),
            up: "",
            body_class: "home",
            sidebar: sidebar(&chapters, None, ""),
            article,
            after: String::new(),
        };
        fs::write(out.join("index.html"), page.html(&site))?;
    }

    // Figures, sample recordings and the site's own assets are copied as they are.
    copy_dir(&root.join("figures"), &out.join("figures"), &["svg"])?;
    copy_dir(&root.join("samples"), &out.join("samples"), &["wav"])?;
    copy_dir(&root.join("site/assets"), &out.join("assets"), &["css", "js", "svg"])?;

    // The interactive pages link back to the Markdown chapters and to source
    // files; on the site those links go to the pages and to the repository.
    fs::create_dir_all(out.join("interactive"))?;
    for entry in fs::read_dir(root.join("interactive"))? {
        let path = entry?.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
        match path.extension().and_then(|e| e.to_str()) {
            Some("html") => {
                let html = fs::read_to_string(&path)?;
                let html = relink(&html, &site).replacen("</body>", &format!("{}</body>", site.counter_unless_embedded()), 1);
                fs::write(out.join("interactive").join(&name), html)?;
            }
            Some("css" | "js") => {
                fs::copy(&path, out.join("interactive").join(&name))?;
            }
            _ => {}
        }
    }

    // The listening page runs the recognizer itself, compiled to WebAssembly.
    // It needs the module and the trained weights next to it.
    match build_wasm(&root) {
        Ok(wasm) => {
            fs::copy(wasm, out.join("interactive/listen.wasm"))?;
            fs::copy(root.join("models/recognizer.bin"), out.join("interactive/recognizer.bin"))?;
        }
        Err(why) => {
            println!("the listening page will say the recognizer is missing: {why}");
            println!("  (to fix: rustup target add wasm32-unknown-unknown)");
        }
    }

    // GitHub Pages runs Jekyll over the files unless this file is present.
    fs::write(out.join(".nojekyll"), "")?;

    println!("wrote {} chapters and the front page to {}", chapters.len(), out.display());
    if site.repo.is_none() {
        println!("SITE_REPO_URL is not set: links to source files are plain text");
    }
    Ok(())
}

/// Compiles `web/` (the recognizer behind a few plain functions) for the
/// browser and returns the path of the module.
fn build_wasm(root: &Path) -> Result<PathBuf, String> {
    let target = "wasm32-unknown-unknown";
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let status = std::process::Command::new(cargo)
        .args(["build", "--release", "--target", target, "--manifest-path"])
        .arg(root.join("web/Cargo.toml"))
        .status()
        .map_err(|e| format!("could not run cargo: {e}"))?;
    if !status.success() {
        return Err(format!("building web/ for {target} failed"));
    }
    let wasm = root.join("web/target").join(target).join("release/speech_web.wasm");
    if wasm.exists() {
        Ok(wasm)
    } else {
        Err(format!("{} was not produced", wasm.display()))
    }
}

fn pager_link(class: &str, label: &str, href: &str, chapter: &Chapter) -> String {
    let overline = if class == "prev" { format!("← {label}") } else { format!("{label} →") };
    format!(
        "<a class=\"pager-link {class}\" href=\"{href}\"><span class=\"overline\">{overline} · Chapter {}</span><span class=\"pager-title\">{}</span></a>",
        chapter.number,
        escape(&chapter.title)
    )
}

/// The README's title is in Title Case for GitHub; the book's name is not.
fn sentence_case_title(title: &str) -> String {
    if title.eq_ignore_ascii_case(BOOK) {
        BOOK.to_string()
    } else {
        title.to_string()
    }
}

/// Splits a Markdown file into its title (the first heading) and the rest.
fn split_title(source: &str) -> (String, String) {
    let mut title = String::new();
    let mut rest = Vec::new();
    for line in source.lines() {
        if title.is_empty() && line.starts_with('#') {
            title = line.trim_start_matches('#').trim().to_string();
        } else {
            rest.push(line);
        }
    }
    (title, rest.join("\n"))
}

fn read_chapters(dir: &Path) -> io::Result<Vec<Chapter>> {
    let mut chapters = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let Some(number) = stem.split('-').next().and_then(|n| n.parse().ok()) else { continue };
        let source = fs::read_to_string(&path)?;
        let (title, body) = split_title(&source);
        // Reading time at 220 words a minute; listings count, because the
        // reader is meant to read them.
        let minutes = (source.split_whitespace().count() / 220).max(1);
        chapters.push(Chapter { number, stem, title, body, minutes });
    }
    chapters.sort_by_key(|c| c.number);
    Ok(chapters)
}

/// The table of contents down the side of every page. On a chapter page the
/// chapter's own sections are listed under it.
fn sidebar(chapters: &[Chapter], current: Option<(usize, &Vec<markdown::Heading>)>, up: &str) -> String {
    let mut html = String::new();
    for (roman, name, range) in PARTS {
        html.push_str(&format!("<p class=\"overline\">Part {roman} · {}</p>\n<ol class=\"toc\">\n", escape(name)));
        for chapter in chapters.iter().filter(|c| range.contains(&c.number)) {
            let here = current.as_ref().is_some_and(|(n, _)| *n == chapter.number);
            html.push_str(&format!(
                "<li{}><a href=\"{up}chapters/{}.html\"><span class=\"toc-n\">{:02}</span><span>{}</span></a>",
                if here { " class=\"here\"" } else { "" },
                chapter.stem,
                chapter.number,
                escape(&chapter.title)
            ));
            if let Some((_, headings)) = current.as_ref().filter(|_| here) {
                html.push_str("\n<ol class=\"toc-sections\">\n");
                for h in headings.iter() {
                    html.push_str(&format!("<li><a href=\"#{}\">{}</a></li>\n", h.id, h.html));
                }
                html.push_str("</ol>");
            }
            html.push_str("</li>\n");
        }
        html.push_str("</ol>\n");
    }
    html
}

struct Page {
    title: String,
    description: String,
    /// `"../"` for a page in `chapters/`, `""` for the front page.
    up: &'static str,
    body_class: &'static str,
    sidebar: String,
    article: String,
    /// What follows the article inside `<main>`: the previous/next links.
    after: String,
}

impl Page {
    fn html(&self, site: &Site) -> String {
        let up = self.up;
        let code_link = match &site.repo {
            Some(repo) => format!("<a href=\"{}\">Code</a>", escape(repo)),
            None => String::new(),
        };
        format!(
            r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<script>
// Before first paint: use the theme the reader chose, if they chose one.
// The interactive pages read and write the same key.
try {{
  var t = localStorage.getItem('sr-theme');
  if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
}} catch (e) {{}}
</script>
<link rel="stylesheet" href="{up}assets/site.css">
<link rel="icon" href="{up}assets/icon.svg" type="image/svg+xml">
</head>
<body class="{body_class}">
<a class="skip" href="#content">Skip to the text</a>
<header class="topbar">
  <button class="icon-button menu-button" type="button" aria-label="Contents" aria-expanded="false" aria-controls="sidebar">
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.25" stroke-linecap="round" aria-hidden="true"><path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11"/></svg>
  </button>
  <a class="brand" href="{up}index.html">Building a Speech Recognizer <span>from Scratch</span></a>
  <nav class="topnav" aria-label="Site">
    <a class="hide-narrow" href="{up}index.html#how-to-read-this-book">Contents</a>
    <a class="hide-narrow" href="{up}interactive/index.html">Interactives</a>
    {code_link}
    <button class="icon-button theme-toggle" type="button" aria-label="Switch theme"></button>
  </nav>
</header>
<div class="layout">
  <nav class="sidebar" id="sidebar" aria-label="Chapters">
{sidebar}
  </nav>
  <main id="content">
    <article class="prose">
{article}
    </article>
{after}
    <footer class="site-footer">
      <span><em>{book}</em>. From sound waves to words, in Rust.</span>
      <span>This page is generated from the Markdown chapters by a small Rust program in the same repository.</span>
    </footer>
  </main>
</div>
<script src="{up}assets/site.js"></script>
{counter}</body>
</html>
"##,
            title = escape(&self.title),
            description = escape(&self.description),
            body_class = self.body_class,
            sidebar = self.sidebar,
            article = self.article,
            after = self.after,
            book = BOOK,
            counter = site.counter(),
        )
    }
}

/// Rewrites the links of an interactive page for the site: chapters are
/// `.html` pages here, and source files are in the repository.
fn relink(html: &str, site: &Site) -> String {
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    let needle = "href=\"";
    while let Some(at) = rest.find(needle) {
        let start = at + needle.len();
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let end = tail.find('"').unwrap_or(tail.len());
        let url = &tail[..end];
        // Links between the interactive pages themselves are already right.
        let leaves_the_directory = url.starts_with("../") || url.ends_with(".md");
        match site.resolve(url, "interactive") {
            _ if !leaves_the_directory => out.push_str(url),
            Some(target) => out.push_str(&escape(&target)),
            // No repository to point at: leave the reader on the page.
            None => out.push('#'),
        }
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

fn copy_dir(from: &Path, to: &Path, extensions: &[&str]) -> io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let path = entry?.path();
        let wanted = path.extension().and_then(|e| e.to_str()).is_some_and(|e| extensions.contains(&e));
        if path.is_file() && wanted {
            fs::copy(&path, to.join(path.file_name().expect("a file has a name")))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(analytics: Option<&str>) -> Site {
        Site { repo: None, branch: "main".to_string(), analytics: analytics_token(analytics.map(str::to_string)) }
    }

    #[test]
    fn no_token_means_no_counter() {
        assert_eq!(site(None).counter(), "");
        assert_eq!(site(None).counter_unless_embedded(), "");
        assert_eq!(site(Some("  ")).counter(), "");
    }

    #[test]
    fn a_token_that_is_not_plain_is_refused() {
        assert_eq!(site(Some("abc\"'><script>")).counter(), "");
        assert_eq!(site(Some("abc def")).counter(), "");
    }

    #[test]
    fn the_counter_carries_the_token() {
        let with = site(Some(" 0123abcd\n"));
        assert!(with.counter().contains(r#"data-cf-beacon='{"token": "0123abcd"}'"#));
        assert!(with.counter().contains(BEACON));
        let embedded = with.counter_unless_embedded();
        assert!(embedded.contains(r#"'{"token": "0123abcd"}'"#));
        assert!(embedded.contains("embed"));
    }
}
