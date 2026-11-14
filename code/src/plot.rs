//! Figures without a plotting library: SVG files and terminal sketches.
//!
//! Every figure in the book that shows data was drawn by this module, from
//! the same numbers our recognizer works with. There are three kinds of
//! chart, each built with chained method calls and each able to save itself:
//!
//! ```no_run
//! use speech::plot::{LinePlot, Heatmap, BarChart, Figure, Orientation};
//! # let (xs, ys) = (vec![0.0f32], vec![0.0f32]);
//! # let spectrogram = speech::mat::Mat::zeros(2, 2);
//! LinePlot::new("A waveform")
//!     .x_label("time (s)")
//!     .y_label("amplitude")
//!     .line("seven", &xs, &ys)
//!     .save("../figures/example.svg")?;
//!
//! // Several charts stacked in one figure, sharing the time axis:
//! Figure::new("Waveform and spectrogram")
//!     .panel(LinePlot::new("waveform").y_label("amplitude").line("", &xs, &ys).x_range(0.0, 1.0))
//!     .panel(Heatmap::new("spectrogram", &spectrogram).orientation(Orientation::TimeOnX).x_range(0.0, 1.0).x_label("time (s)"))
//!     .share_x()
//!     .save("../figures/example2.svg")?;
//!
//! BarChart::new("What the network thinks", &["yes", "no"], &[0.9, 0.1]).save("../figures/example3.svg")?;
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! An SVG file is text: a list of shapes (`<line>`, `<rect>`, `<text>`...)
//! with coordinates in pixels, `y` growing downwards. Drawing a chart is
//! therefore mostly arithmetic (mapping data values to pixel positions) and
//! string formatting. The look follows the book's house style: warm paper,
//! near-black ink, thin lines, one green accent.
//!
//! For quick looks without leaving the terminal there are [`waveform_ascii`],
//! [`heatmap_ascii`] and [`sparkline`], which draw with Unicode block
//! characters.

use crate::mat::Mat;
use std::fmt::Write as _;
use std::io;
use std::path::Path;

// ---------------------------------------------------------------------------
// House style
// ---------------------------------------------------------------------------

/// Background of every figure.
pub const PAPER: &str = "#F5F3EF";
/// Text and primary strokes.
pub const INK: &str = "#191615";
const SANS: &str = "Geist, Inter, system-ui, -apple-system, 'Segoe UI', sans-serif";
const MONO: &str = "'Geist Mono', ui-monospace, SFMono-Regular, Menlo, monospace";

/// Width of every figure in SVG pixels.
const WIDTH: f32 = 800.0;
/// Space left of the plot area, for tick labels and the y-axis label.
const MARGIN_LEFT: f32 = 68.0;
/// Space right of the plot area.
const MARGIN_RIGHT: f32 = 24.0;
/// Space right of the plot area when a panel has a color bar.
const MARGIN_RIGHT_COLORBAR: f32 = 84.0;

/// The colors a chart element can have. A small fixed palette on purpose:
/// green is "the thing this figure is about", amber is the second series,
/// red is reserved for errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    /// Near-black, `#191615`.
    Ink,
    /// Ink at 70% opacity: secondary text and lines.
    Ink70,
    /// Ink at 48% opacity: captions, reference lines.
    Ink48,
    /// Ink at 30% opacity: axes, de-emphasized series.
    Ink30,
    /// The accent, `#047857`.
    Green,
    /// The second series, `#B45309`.
    Amber,
    /// Errors and wrong answers only, `#B42318`.
    Red,
    /// The background color, for marks drawn on top of dark cells.
    Paper,
}

impl Color {
    /// The CSS color string.
    pub fn css(self) -> &'static str {
        match self {
            Color::Ink => INK,
            Color::Ink70 => "rgba(25,22,21,.70)",
            Color::Ink48 => "rgba(25,22,21,.48)",
            Color::Ink30 => "rgba(25,22,21,.30)",
            Color::Green => "#047857",
            Color::Amber => "#B45309",
            Color::Red => "#B42318",
            Color::Paper => PAPER,
        }
    }

    /// The color as it looks on paper, but opaque. Filled shapes use this so
    /// that grid lines do not show through them.
    fn solid(self) -> &'static str {
        match self {
            Color::Ink70 => "#5B5957",
            Color::Ink48 => "#8B8986",
            Color::Ink30 => "#B3B1AE",
            other => other.css(),
        }
    }

    /// The same color as a translucent fill, for shaded regions.
    fn wash(self) -> &'static str {
        match self {
            Color::Green => "rgba(4,120,87,.10)",
            Color::Amber => "rgba(180,83,9,.10)",
            Color::Red => "rgba(180,35,24,.10)",
            _ => "rgba(25,22,21,.07)",
        }
    }
}

/// Colors handed out to series that did not ask for one, in order.
const SERIES_COLORS: [Color; 4] = [Color::Green, Color::Amber, Color::Ink70, Color::Ink30];

const AXIS: &str = "rgba(25,22,21,.30)";
const HAIRLINE: &str = "rgba(25,22,21,.10)";

/// The heatmap color ramp at `t` in `[0, 1]`: paper, sage, deep green, ink.
/// A single hue that only gets darker, so it reads correctly in grayscale
/// and for color-blind readers (unlike a rainbow).
pub fn ramp(t: f32) -> [u8; 3] {
    const STOPS: [(f32, [f32; 3]); 4] =
        [(0.0, [245.0, 243.0, 239.0]), (0.45, [159.0, 184.0, 169.0]), (0.85, [4.0, 84.0, 62.0]), (1.0, [25.0, 22.0, 21.0])];
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    for pair in STOPS.windows(2) {
        let ((t0, c0), (t1, c1)) = (pair[0], pair[1]);
        if t <= t1 {
            let f = (t - t0) / (t1 - t0);
            return [0, 1, 2].map(|i| (c0[i] + f * (c1[i] - c0[i])).round() as u8);
        }
    }
    [25, 22, 21]
}

fn ramp_css(t: f32) -> String {
    let [r, g, b] = ramp(t);
    format!("#{r:02X}{g:02X}{b:02X}")
}

// ---------------------------------------------------------------------------
// Small SVG helpers
// ---------------------------------------------------------------------------

/// Escape the characters that have a meaning in XML.
pub fn escape_xml(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
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

/// Format a pixel coordinate: two decimals at most, no trailing zeros.
fn px(v: f32) -> String {
    let s = format!("{v:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" {
        "0".to_string()
    } else {
        s.to_string()
    }
}

/// The plot area of a panel, in pixels.
#[derive(Clone, Copy, Debug)]
struct Area {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

/// A linear map from data values to pixel positions.
#[derive(Clone, Copy, Debug)]
struct Scale {
    lo: f32,
    hi: f32,
    px_lo: f32,
    px_hi: f32,
}

impl Scale {
    fn at(&self, v: f32) -> f32 {
        let span = self.hi - self.lo;
        let t = if span == 0.0 { 0.5 } else { (v - self.lo) / span };
        self.px_lo + t * (self.px_hi - self.px_lo)
    }
}

/// The text styles of the house style.
#[derive(Clone, Copy)]
enum TextStyle {
    /// Figure title: sans 15px, medium weight.
    Title,
    /// Axis labels, legend entries, annotations: sans 12px.
    Label,
    /// Axis tick values: mono 10px, faint.
    Tick,
    /// Small-caps style labels: mono 10px, uppercase, letter-spaced.
    Overline,
    /// Category names along an axis: mono 11px.
    Category,
}

#[derive(Clone, Copy)]
enum Anchor {
    Start,
    Middle,
    End,
}

/// Rough width of a string in pixels. We cannot measure text without a font
/// engine, so we estimate from the character count; good enough to keep
/// labels from colliding.
fn text_width(text: &str, style: TextStyle) -> f32 {
    let per_char = match style {
        TextStyle::Title => 8.0,
        TextStyle::Label => 6.7,
        TextStyle::Tick => 6.1,
        TextStyle::Overline => 6.9,
        TextStyle::Category => 6.7,
    };
    text.chars().count() as f32 * per_char
}

/// Append a `<text>` element at `(x, y)`. `rotate` turns the text about that point.
fn text(svg: &mut String, (x, y): (f32, f32), content: &str, style: TextStyle, anchor: Anchor, fill: &str, rotate: f32) {
    let (family, size, extra) = match style {
        TextStyle::Title => (SANS, 15, " font-weight=\"500\""),
        TextStyle::Label => (SANS, 12, ""),
        TextStyle::Tick => (MONO, 10, ""),
        TextStyle::Overline => (MONO, 10, " letter-spacing=\".08em\""),
        TextStyle::Category => (MONO, 11, ""),
    };
    let anchor = match anchor {
        Anchor::Start => "",
        Anchor::Middle => " text-anchor=\"middle\"",
        Anchor::End => " text-anchor=\"end\"",
    };
    let transform =
        if rotate == 0.0 { String::new() } else { format!(" transform=\"rotate({} {} {})\"", px(rotate), px(x), px(y)) };
    let content = if matches!(style, TextStyle::Overline) { content.to_uppercase() } else { content.to_string() };
    let _ = writeln!(
        svg,
        "<text x=\"{}\" y=\"{}\" font-family=\"{family}\" font-size=\"{size}\"{extra}{anchor} fill=\"{fill}\"{transform}>{}</text>",
        px(x),
        px(y),
        escape_xml(&content)
    );
}

fn line(svg: &mut String, (x1, y1): (f32, f32), (x2, y2): (f32, f32), stroke: &str, width: f32, dash: &str) {
    let dash = if dash.is_empty() { String::new() } else { format!(" stroke-dasharray=\"{dash}\"") };
    let _ = writeln!(
        svg,
        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{stroke}\" stroke-width=\"{}\" stroke-linecap=\"round\"{dash}/>",
        px(x1),
        px(y1),
        px(x2),
        px(y2),
        px(width)
    );
}

// ---------------------------------------------------------------------------
// Ticks
// ---------------------------------------------------------------------------

/// Pick "nice" tick positions covering `lo..=hi`: multiples of 1, 2 or 5
/// times a power of ten, about `target` of them. Returns the ticks and the
/// number of decimals needed to print them.
pub fn nice_ticks(lo: f32, hi: f32, target: usize) -> (Vec<f32>, usize) {
    let (lo, hi) = (lo.min(hi) as f64, lo.max(hi) as f64);
    let span = hi - lo;
    if span <= 0.0 || !span.is_finite() {
        return (vec![lo as f32], 0);
    }
    // The ideal step, moved to the nearest of 1, 2, 5 x 10^k ("nearest" as
    // a ratio: 3 is closer to 2 than to 5).
    let raw_step = span / target.max(1) as f64;
    let magnitude = 10f64.powf(raw_step.log10().floor());
    let candidates = [1.0, 2.0, 5.0, 10.0].map(|m| m * magnitude);
    let distance = |step: f64| (step / raw_step).ln().abs();
    let step = candidates.into_iter().min_by(|a, b| distance(*a).total_cmp(&distance(*b))).unwrap_or(magnitude);

    let decimals = (-step.log10().floor()).max(0.0) as usize;
    let mut ticks = Vec::new();
    let tolerance = step * 1e-4;
    let mut i = ((lo - tolerance) / step).ceil();
    while i * step <= hi + tolerance {
        ticks.push((i * step) as f32);
        i += 1.0;
    }
    (ticks, decimals)
}

/// Print a tick value with a real minus sign and no "-0".
fn tick_label(value: f32, decimals: usize) -> String {
    let s = format!("{value:.decimals$}");
    let is_zero = s.trim_start_matches('-').chars().all(|c| c == '0' || c == '.');
    match (is_zero, s.strip_prefix('-')) {
        (true, Some(rest)) => rest.to_string(),
        (false, Some(rest)) => format!("\u{2212}{rest}"),
        _ => s,
    }
}

// ---------------------------------------------------------------------------
// Line plots
// ---------------------------------------------------------------------------

/// One curve (or cloud of dots) on a [`LinePlot`].
#[derive(Clone, Debug)]
pub struct Series {
    label: String,
    xs: Vec<f32>,
    ys: Vec<f32>,
    color: Option<Color>,
    width: f32,
    dashed: bool,
    dots: bool,
    connect: bool,
}

impl Series {
    /// A curve through the points `(xs[i], ys[i])`. An empty `label` keeps
    /// the series out of the legend.
    pub fn new(label: &str, xs: &[f32], ys: &[f32]) -> Self {
        assert_eq!(xs.len(), ys.len(), "series \"{label}\": {} x values but {} y values", xs.len(), ys.len());
        Series {
            label: label.to_string(),
            xs: xs.to_vec(),
            ys: ys.to_vec(),
            color: None,
            width: 1.5,
            dashed: false,
            dots: false,
            connect: true,
        }
    }

    /// A curve through `ys`, with x = 0, 1, 2, ...
    pub fn from_y(label: &str, ys: &[f32]) -> Self {
        let xs: Vec<f32> = (0..ys.len()).map(|i| i as f32).collect();
        Series::new(label, &xs, ys)
    }

    /// Set the color (otherwise green, amber, ... in order of appearance).
    pub fn color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    /// Set the stroke width in pixels (default 1.5).
    pub fn width(mut self, width: f32) -> Self {
        self.width = width;
        self
    }

    /// Draw the line dashed.
    pub fn dashed(mut self) -> Self {
        self.dashed = true;
        self
    }

    /// Mark every data point with a dot (for showing individual samples).
    pub fn dots(mut self) -> Self {
        self.dots = true;
        self
    }

    /// Draw only the dots, without the connecting line.
    pub fn dots_only(mut self) -> Self {
        self.dots = true;
        self.connect = false;
        self
    }
}

#[derive(Clone, Debug)]
struct Span {
    from: f32,
    to: f32,
    label: String,
    color: Color,
}

#[derive(Clone, Debug)]
struct Marker {
    at: f32,
    label: String,
    color: Color,
}

#[derive(Clone, Debug)]
struct Note {
    x: f32,
    y: f32,
    label: String,
}

/// A chart of one or more curves over shared axes.
#[derive(Clone, Debug)]
pub struct LinePlot {
    title: String,
    x_label: String,
    y_label: String,
    series: Vec<Series>,
    x_range: Option<(f32, f32)>,
    y_range: Option<(f32, f32)>,
    spans: Vec<Span>,
    vlines: Vec<Marker>,
    hlines: Vec<Marker>,
    notes: Vec<Note>,
    legend: Option<bool>,
    height: f32,
}

impl LinePlot {
    /// An empty plot. The title is the figure's title when the plot is saved
    /// on its own, and a small label above the panel inside a [`Figure`].
    pub fn new(title: &str) -> Self {
        LinePlot {
            title: title.to_string(),
            x_label: String::new(),
            y_label: String::new(),
            series: Vec::new(),
            x_range: None,
            y_range: None,
            spans: Vec::new(),
            vlines: Vec::new(),
            hlines: Vec::new(),
            notes: Vec::new(),
            legend: None,
            height: 220.0,
        }
    }

    /// Label under the x axis. Include the unit: `"time (s)"`.
    pub fn x_label(mut self, label: &str) -> Self {
        self.x_label = label.to_string();
        self
    }

    /// Label beside the y axis. Include the unit.
    pub fn y_label(mut self, label: &str) -> Self {
        self.y_label = label.to_string();
        self
    }

    /// Add a curve through `(xs[i], ys[i])` in the next free color.
    pub fn line(self, label: &str, xs: &[f32], ys: &[f32]) -> Self {
        self.series(Series::new(label, xs, ys))
    }

    /// Add a curve through `ys` with x = 0, 1, 2, ...
    pub fn line_y(self, label: &str, ys: &[f32]) -> Self {
        self.series(Series::from_y(label, ys))
    }

    /// Add a fully configured [`Series`].
    pub fn series(mut self, series: Series) -> Self {
        self.series.push(series);
        self
    }

    /// Fix the x axis to `lo..hi` instead of fitting it to the data.
    pub fn x_range(mut self, lo: f32, hi: f32) -> Self {
        self.x_range = Some((lo, hi));
        self
    }

    /// Fix the y axis to `lo..hi` instead of fitting it to the data.
    pub fn y_range(mut self, lo: f32, hi: f32) -> Self {
        self.y_range = Some((lo, hi));
        self
    }

    /// Shade the region between two x values, in green, with a label.
    pub fn span(self, from: f32, to: f32, label: &str) -> Self {
        self.span_colored(from, to, label, Color::Green)
    }

    /// Shade the region between two x values in a color of your choice.
    pub fn span_colored(mut self, from: f32, to: f32, label: &str, color: Color) -> Self {
        self.spans.push(Span { from: from.min(to), to: from.max(to), label: label.to_string(), color });
        self
    }

    /// A dashed vertical marker at `x`, with a label at its top.
    pub fn vline(mut self, x: f32, label: &str) -> Self {
        self.vlines.push(Marker { at: x, label: label.to_string(), color: Color::Ink48 });
        self
    }

    /// A dashed horizontal marker at `y` (a threshold, say), labeled at its
    /// right end.
    pub fn hline(mut self, y: f32, label: &str) -> Self {
        self.hlines.push(Marker { at: y, label: label.to_string(), color: Color::Ink48 });
        self
    }

    /// Write a short note centered just above the data point `(x, y)`.
    pub fn annotate(mut self, x: f32, y: f32, label: &str) -> Self {
        self.notes.push(Note { x, y, label: label.to_string() });
        self
    }

    /// Force the legend on or off. By default it appears when at least two
    /// series have a label.
    pub fn legend(mut self, show: bool) -> Self {
        self.legend = Some(show);
        self
    }

    /// Height of the plot area in pixels (default 220).
    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// Render as a standalone SVG document.
    pub fn to_svg(&self) -> String {
        Figure::new(&self.title).panel(self.clone()).to_svg()
    }

    /// Render and write to `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        Figure::new(&self.title).panel(self.clone()).save(path)
    }

    /// The x and y ranges that will be drawn: the fixed ones if given,
    /// otherwise fitted to the data (y with a little headroom).
    fn ranges(&self) -> ((f32, f32), (f32, f32)) {
        let finite = |values: &mut dyn Iterator<Item = f32>| {
            values.filter(|v| v.is_finite()).fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)))
        };
        let x = self.x_range.unwrap_or_else(|| {
            let (lo, hi) = finite(&mut self.series.iter().flat_map(|s| s.xs.iter().copied()));
            if lo > hi {
                (0.0, 1.0)
            } else if lo == hi {
                (lo - 0.5, hi + 0.5)
            } else {
                (lo, hi)
            }
        });
        let y = self.y_range.unwrap_or_else(|| {
            let data = self.series.iter().flat_map(|s| s.ys.iter().copied());
            let (lo, hi) = finite(&mut data.chain(self.hlines.iter().map(|h| h.at)));
            if lo > hi {
                (0.0, 1.0)
            } else if lo == hi {
                (lo - 0.5, hi + 0.5)
            } else {
                let pad = 0.06 * (hi - lo);
                (lo - pad, hi + pad)
            }
        });
        (x, y)
    }

    fn legend_entries(&self) -> Vec<(&str, Color, bool)> {
        let labeled: Vec<_> = self
            .series
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.label.is_empty())
            .map(|(i, s)| (s.label.as_str(), s.color.unwrap_or(SERIES_COLORS[i % SERIES_COLORS.len()]), s.dashed))
            .collect();
        match self.legend {
            Some(false) => Vec::new(),
            Some(true) => labeled,
            None if labeled.len() >= 2 => labeled,
            None => Vec::new(),
        }
    }

    fn header_height(&self, show_title: bool) -> f32 {
        if (show_title && !self.title.is_empty()) || !self.legend_entries().is_empty() {
            22.0
        } else {
            4.0
        }
    }

    fn draw(&self, svg: &mut String, area: Area, env: &Env) {
        let ((x_lo, x_hi), (y_lo, y_hi)) = self.ranges();
        let xs = Scale { lo: x_lo, hi: x_hi, px_lo: area.x, px_hi: area.x + area.w };
        let ys = Scale { lo: y_lo, hi: y_hi, px_lo: area.y + area.h, px_hi: area.y };
        let (bottom, right) = (area.y + area.h, area.x + area.w);

        // Header: panel label on the left, legend on the right.
        if env.show_title && !self.title.is_empty() {
            text(svg, (area.x, area.y - 9.0), &self.title, TextStyle::Overline, Anchor::Start, Color::Ink48.css(), 0.0);
        }
        let mut legend_x = right;
        for (label, color, dashed) in self.legend_entries().into_iter().rev() {
            legend_x -= text_width(label, TextStyle::Label);
            text(svg, (legend_x, area.y - 8.0), label, TextStyle::Label, Anchor::Start, Color::Ink70.css(), 0.0);
            legend_x -= 24.0;
            line(
                svg,
                (legend_x, area.y - 12.0),
                (legend_x + 18.0, area.y - 12.0),
                color.css(),
                2.0,
                if dashed { "4 3" } else { "" },
            );
            legend_x -= 16.0;
        }

        // Shaded spans go underneath everything else.
        for span in &self.spans {
            let (a, b) = (xs.at(span.from).clamp(area.x, right), xs.at(span.to).clamp(area.x, right));
            let _ = writeln!(
                svg,
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\"/>",
                px(a),
                px(area.y),
                px(b - a),
                px(area.h),
                span.color.wash()
            );
            if !span.label.is_empty() {
                text(svg, (a + 6.0, area.y + 14.0), &span.label, TextStyle::Overline, Anchor::Start, span.color.css(), 0.0);
            }
        }

        // Horizontal grid lines and y tick labels.
        let (y_ticks, y_decimals) = nice_ticks(y_lo, y_hi, (area.h / 45.0).round().max(2.0) as usize);
        for &tick in &y_ticks {
            let y = ys.at(tick);
            line(svg, (area.x, y), (right, y), HAIRLINE, 1.0, "");
            text(
                svg,
                (area.x - 8.0, y + 3.5),
                &tick_label(tick, y_decimals),
                TextStyle::Tick,
                Anchor::End,
                Color::Ink48.css(),
                0.0,
            );
        }
        // Axes: a left and a bottom line, nothing else.
        line(svg, (area.x, area.y), (area.x, bottom), AXIS, 1.0, "");
        line(svg, (area.x, bottom), (right, bottom), AXIS, 1.0, "");
        let (x_ticks, x_decimals) = nice_ticks(x_lo, x_hi, 8);
        for &tick in &x_ticks {
            let x = xs.at(tick);
            line(svg, (x, bottom), (x, bottom + 4.0), AXIS, 1.0, "");
            if env.show_x_text {
                text(
                    svg,
                    (x, bottom + 16.0),
                    &tick_label(tick, x_decimals),
                    TextStyle::Tick,
                    Anchor::Middle,
                    Color::Ink48.css(),
                    0.0,
                );
            }
        }
        if env.show_x_text && !self.x_label.is_empty() {
            text(
                svg,
                (area.x + area.w / 2.0, bottom + 36.0),
                &self.x_label,
                TextStyle::Label,
                Anchor::Middle,
                Color::Ink70.css(),
                0.0,
            );
        }
        if !self.y_label.is_empty() {
            text(svg, (20.0, area.y + area.h / 2.0), &self.y_label, TextStyle::Label, Anchor::Middle, Color::Ink70.css(), -90.0);
        }

        // Horizontal markers (thresholds).
        for marker in &self.hlines {
            let y = ys.at(marker.at);
            line(svg, (area.x, y), (right, y), marker.color.css(), 1.0, "4 3");
            if !marker.label.is_empty() {
                text(svg, (right - 4.0, y - 5.0), &marker.label, TextStyle::Overline, Anchor::End, Color::Ink70.css(), 0.0);
            }
        }

        // The data, clipped to the plot area.
        let _ = writeln!(
            svg,
            "<clipPath id=\"clip{}\"><rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"/></clipPath>\n<g clip-path=\"url(#clip{})\">",
            env.id,
            px(area.x),
            px(area.y - 3.0),
            px(area.w + 3.0),
            px(area.h + 6.0),
            env.id
        );
        for (i, series) in self.series.iter().enumerate() {
            let color = series.color.unwrap_or(SERIES_COLORS[i % SERIES_COLORS.len()]).css();
            let points: Vec<(f32, f32)> = series
                .xs
                .iter()
                .zip(&series.ys)
                .filter(|(x, y)| x.is_finite() && y.is_finite())
                .map(|(&x, &y)| (xs.at(x), ys.at(y)))
                .collect();
            if series.connect && points.len() >= 2 {
                // Thousands of samples squeezed into a few hundred pixels:
                // keep only the lowest and highest point of every pixel column.
                let drawn =
                    if !series.dots && points.len() as f32 > 4.0 * area.w { min_max_per_pixel(&points) } else { points.clone() };
                let mut path = String::with_capacity(drawn.len() * 12);
                for (x, y) in drawn {
                    let _ = write!(path, "{},{} ", px(x), px(y));
                }
                let dash = if series.dashed { " stroke-dasharray=\"4 3\"" } else { "" };
                let _ = writeln!(
                    svg,
                    "<polyline points=\"{}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"{}\" stroke-linecap=\"round\" stroke-linejoin=\"round\"{dash}/>",
                    path.trim_end(),
                    px(series.width)
                );
            }
            if series.dots {
                for &(x, y) in &points {
                    let _ = writeln!(
                        svg,
                        "<circle cx=\"{}\" cy=\"{}\" r=\"2.4\" fill=\"{color}\" stroke=\"{PAPER}\" stroke-width=\"0.75\"/>",
                        px(x),
                        px(y)
                    );
                }
            }
        }
        svg.push_str("</g>\n");

        // Vertical markers and notes sit on top of the data.
        for marker in &self.vlines {
            let x = xs.at(marker.at);
            line(svg, (x, area.y), (x, bottom), marker.color.css(), 1.0, "4 3");
            if !marker.label.is_empty() {
                let fits_right = x + 6.0 + text_width(&marker.label, TextStyle::Overline) < right;
                let (label_x, anchor) = if fits_right { (x + 6.0, Anchor::Start) } else { (x - 6.0, Anchor::End) };
                text(svg, (label_x, area.y + 14.0), &marker.label, TextStyle::Overline, anchor, Color::Ink70.css(), 0.0);
            }
        }
        for note in &self.notes {
            text(
                svg,
                (xs.at(note.x), ys.at(note.y) - 8.0),
                &note.label,
                TextStyle::Label,
                Anchor::Middle,
                Color::Ink70.css(),
                0.0,
            );
        }
    }
}

/// Reduce a long polyline to at most four points per pixel column (first,
/// lowest, highest, last), which looks identical at this resolution and
/// keeps the file small.
fn min_max_per_pixel(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < points.len() {
        let column = points[start].0.floor();
        let mut end = start;
        while end < points.len() && points[end].0.floor() == column {
            end += 1;
        }
        let bucket = &points[start..end];
        let lowest = (0..bucket.len()).min_by(|&a, &b| bucket[a].1.total_cmp(&bucket[b].1)).unwrap_or(0);
        let highest = (0..bucket.len()).max_by(|&a, &b| bucket[a].1.total_cmp(&bucket[b].1)).unwrap_or(0);
        let mut keep = vec![0, lowest, highest, bucket.len() - 1];
        keep.sort_unstable();
        keep.dedup();
        out.extend(keep.into_iter().map(|i| bucket[i]));
        start = end;
    }
    out
}

// ---------------------------------------------------------------------------
// Heatmaps
// ---------------------------------------------------------------------------

/// Which way round a matrix is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// As you would write the matrix down: row 0 at the top, column 0 on the
    /// left. Cells are square. Good for lattices and tables.
    Matrix,
    /// Like `Matrix`, but with row 0 at the *bottom*, so both indices grow
    /// away from the bottom-left corner like the axes of a graph. The usual
    /// way to draw a DTW cost grid.
    MatrixFlipped,
    /// Rows run left to right and columns run bottom to top. Our feature
    /// matrices have one row per frame, so this puts time on the x axis and
    /// frequency on the y axis, low frequencies at the bottom: a spectrogram.
    /// Cells are stretched to fill the plot area.
    TimeOnX,
}

/// A matrix drawn as a grid of colored cells: spectrograms, DTW cost grids,
/// probability lattices.
#[derive(Clone, Debug)]
pub struct Heatmap {
    title: String,
    values: Mat,
    orientation: Orientation,
    x_label: String,
    y_label: String,
    x_range: Option<(f32, f32)>,
    y_range: Option<(f32, f32)>,
    value_range: Option<(f32, f32)>,
    colorbar: Option<String>,
    paths: Vec<(Vec<(usize, usize)>, Color)>,
    outlines: Vec<(usize, usize, Color)>,
    cell_text: Option<usize>,
    row_labels: Vec<String>,
    col_labels: Vec<String>,
    height: Option<f32>,
}

/// Grids with more cells than this are embedded as one small image instead
/// of one `<rect>` per cell (a spectrogram has tens of thousands of cells).
const MAX_RECT_CELLS: usize = 2000;

impl Heatmap {
    /// A heatmap of `values`, drawn as written ([`Orientation::Matrix`]).
    pub fn new(title: &str, values: &Mat) -> Self {
        Heatmap {
            title: title.to_string(),
            values: values.clone(),
            orientation: Orientation::Matrix,
            x_label: String::new(),
            y_label: String::new(),
            x_range: None,
            y_range: None,
            value_range: None,
            colorbar: None,
            paths: Vec::new(),
            outlines: Vec::new(),
            cell_text: None,
            row_labels: Vec::new(),
            col_labels: Vec::new(),
            height: None,
        }
    }

    /// Choose which way round the matrix is drawn.
    pub fn orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Label under the x axis.
    pub fn x_label(mut self, label: &str) -> Self {
        self.x_label = label.to_string();
        self
    }

    /// Label beside the y axis.
    pub fn y_label(mut self, label: &str) -> Self {
        self.y_label = label.to_string();
        self
    }

    /// The data values at the left edge of the first cell and the right edge
    /// of the last one, e.g. `0.0` and the clip's duration in seconds.
    /// Without it the axis shows cell indices.
    pub fn x_range(mut self, lo: f32, hi: f32) -> Self {
        self.x_range = Some((lo, hi));
        self
    }

    /// The data values at the edges of the y axis, e.g. `0.0` and `8000.0`
    /// Hz. `lo` belongs to index 0: the bottom edge for `TimeOnX` and
    /// `MatrixFlipped`, the top edge for `Matrix`.
    pub fn y_range(mut self, lo: f32, hi: f32) -> Self {
        self.y_range = Some((lo, hi));
        self
    }

    /// The values that map to the lightest and the darkest color. Defaults
    /// to the smallest and largest value in the matrix.
    pub fn value_range(mut self, lo: f32, hi: f32) -> Self {
        self.value_range = Some((lo, hi));
        self
    }

    /// Show a color bar on the right, with this label (the unit: `"dB"`).
    pub fn colorbar(mut self, label: &str) -> Self {
        self.colorbar = Some(label.to_string());
        self
    }

    /// Draw a line through the centers of the given `(row, col)` cells, in
    /// order: a DTW warping path, a decoded alignment. The line is drawn
    /// above the cells but below any cell text.
    pub fn path(mut self, cells: &[(usize, usize)], color: Color) -> Self {
        self.paths.push((cells.to_vec(), color));
        self
    }

    /// Draw a box round the cell `(row, col)`.
    pub fn outline(mut self, row: usize, col: usize, color: Color) -> Self {
        self.outlines.push((row, col, color));
        self
    }

    /// Print every cell's value inside it, with this many decimals. Only
    /// sensible for small grids.
    pub fn cell_text(mut self, decimals: usize) -> Self {
        self.cell_text = Some(decimals);
        self
    }

    /// Name the rows (one label per matrix row) instead of numbering them,
    /// e.g. the symbols of a CTC lattice.
    pub fn row_labels(mut self, labels: &[&str]) -> Self {
        assert_eq!(labels.len(), self.values.rows, "{} row labels for {} rows", labels.len(), self.values.rows);
        self.row_labels = labels.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Name the columns (one label per matrix column) instead of numbering them.
    pub fn col_labels(mut self, labels: &[&str]) -> Self {
        assert_eq!(labels.len(), self.values.cols, "{} column labels for {} columns", labels.len(), self.values.cols);
        self.col_labels = labels.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Height of the plot area in pixels. For the square-celled orientations
    /// this is an upper limit.
    pub fn height(mut self, height: f32) -> Self {
        self.height = Some(height);
        self
    }

    /// Render as a standalone SVG document.
    pub fn to_svg(&self) -> String {
        Figure::new(&self.title).panel(self.clone()).to_svg()
    }

    /// Render and write to `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        Figure::new(&self.title).panel(self.clone()).save(path)
    }

    /// Number of cells across and down, as displayed.
    fn grid_size(&self) -> (usize, usize) {
        match self.orientation {
            Orientation::Matrix | Orientation::MatrixFlipped => (self.values.cols, self.values.rows),
            Orientation::TimeOnX => (self.values.rows, self.values.cols),
        }
    }

    /// Where matrix cell `(row, col)` is displayed: (cells from the left,
    /// cells from the top).
    fn grid_position(&self, row: usize, col: usize) -> (usize, usize) {
        match self.orientation {
            Orientation::Matrix => (col, row),
            Orientation::MatrixFlipped => (col, self.values.rows - 1 - row),
            Orientation::TimeOnX => (row, self.values.cols - 1 - col),
        }
    }

    /// Does the y axis grow upwards (index 0 at the bottom)?
    fn y_grows_up(&self) -> bool {
        self.orientation != Orientation::Matrix
    }

    /// Labels for the cells along x and along y (in index order), if any.
    fn axis_categories(&self) -> (&[String], &[String]) {
        match self.orientation {
            Orientation::Matrix | Orientation::MatrixFlipped => (&self.col_labels, &self.row_labels),
            Orientation::TimeOnX => (&self.row_labels, &self.col_labels),
        }
    }

    /// Size of the drawn grid in pixels, given the space available.
    fn drawn_size(&self, available_w: f32) -> (f32, f32) {
        let (nx, ny) = self.grid_size();
        if nx == 0 || ny == 0 {
            return (available_w, self.height.unwrap_or(120.0));
        }
        match self.orientation {
            Orientation::TimeOnX => (available_w, self.height.unwrap_or(200.0)),
            _ => {
                let max_h = self.height.unwrap_or(420.0);
                let cell = (available_w / nx as f32).min(max_h / ny as f32).min(36.0);
                (cell * nx as f32, cell * ny as f32)
            }
        }
    }

    fn limits(&self) -> (f32, f32) {
        self.value_range.unwrap_or_else(|| {
            let finite = self.values.data.iter().copied().filter(|v| v.is_finite());
            let (lo, hi) = finite.fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)));
            if lo > hi {
                (0.0, 1.0)
            } else {
                (lo, hi)
            }
        })
    }

    fn header_height(&self, show_title: bool) -> f32 {
        if show_title && !self.title.is_empty() {
            22.0
        } else {
            4.0
        }
    }

    fn draw(&self, svg: &mut String, area: Area, env: &Env) {
        let (nx, ny) = self.grid_size();
        let (lo, hi) = self.limits();
        let shade = |v: f32| if hi > lo { (v - lo) / (hi - lo) } else { 0.5 };
        let (cell_w, cell_h) = (area.w / nx.max(1) as f32, area.h / ny.max(1) as f32);
        let (bottom, right) = (area.y + area.h, area.x + area.w);

        if env.show_title && !self.title.is_empty() {
            text(svg, (area.x, area.y - 9.0), &self.title, TextStyle::Overline, Anchor::Start, Color::Ink48.css(), 0.0);
        }

        // The cells.
        if nx * ny > MAX_RECT_CELLS {
            // One pixel per cell, stretched to the plot area by the viewer.
            let mut pixels = vec![0u8; nx * ny];
            for row in 0..self.values.rows {
                for col in 0..self.values.cols {
                    let (gx, gy) = self.grid_position(row, col);
                    pixels[gy * nx + gx] = (shade(self.values[(row, col)]).clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
            let palette: Vec<[u8; 3]> = (0..256).map(|i| ramp(i as f32 / 255.0)).collect();
            let png = encode_png(nx, ny, &pixels, &palette);
            let _ = writeln!(
                svg,
                "<image x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" preserveAspectRatio=\"none\" style=\"image-rendering:pixelated\" href=\"data:image/png;base64,{}\"/>",
                px(area.x),
                px(area.y),
                px(area.w),
                px(area.h),
                base64(&png)
            );
        } else {
            // Each cell is drawn a touch too large, so that it overlaps its
            // right and lower neighbors. Cells that merely touch leave
            // hairline gaps where the renderer rounds their edges apart.
            let overlap = 0.4;
            svg.push_str("<g>\n");
            for row in 0..self.values.rows {
                for col in 0..self.values.cols {
                    let (gx, gy) = self.grid_position(row, col);
                    let _ = writeln!(
                        svg,
                        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"{}\"/>",
                        px(area.x + gx as f32 * cell_w),
                        px(area.y + gy as f32 * cell_h),
                        px(cell_w + if gx + 1 < nx { overlap } else { 0.0 }),
                        px(cell_h + if gy + 1 < ny { overlap } else { 0.0 }),
                        ramp_css(shade(self.values[(row, col)]))
                    );
                }
            }
            svg.push_str("</g>\n");
        }
        let center = |row: usize, col: usize| {
            let (gx, gy) = self.grid_position(row, col);
            (area.x + (gx as f32 + 0.5) * cell_w, area.y + (gy as f32 + 0.5) * cell_h)
        };

        // Paths go under the numbers, so the numbers stay readable.
        for (cells, color) in &self.paths {
            let mut points = String::new();
            for &(row, col) in cells {
                let (cx, cy) = center(row, col);
                let _ = write!(points, "{},{} ", px(cx), px(cy));
            }
            let _ = writeln!(
                svg,
                "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>",
                points.trim_end(),
                color.css()
            );
        }

        // Numbers inside the cells, dark on light cells and light on dark.
        if let Some(decimals) = self.cell_text {
            for row in 0..self.values.rows {
                for col in 0..self.values.cols {
                    let value = self.values[(row, col)];
                    let (cx, cy) = center(row, col);
                    let fill = if shade(value) > 0.55 { PAPER } else { INK };
                    let label = if value.is_finite() { tick_label(value, decimals) } else { "\u{221E}".to_string() };
                    text(svg, (cx, cy + 3.5), &label, TextStyle::Tick, Anchor::Middle, fill, 0.0);
                }
            }
        }

        // A thin frame round the grid.
        let _ = writeln!(
            svg,
            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{AXIS}\" stroke-width=\"1\"/>",
            px(area.x),
            px(area.y),
            px(area.w),
            px(area.h)
        );

        // Outlined cells.
        for &(row, col, color) in &self.outlines {
            let (cx, cy) = center(row, col);
            let _ = writeln!(
                svg,
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.75\"/>",
                px(cx - cell_w / 2.0 + 0.5),
                px(cy - cell_h / 2.0 + 0.5),
                px(cell_w - 1.0),
                px(cell_h - 1.0),
                color.css()
            );
        }

        // Axes: category names if given, else numbers.
        let (x_categories, y_categories) = self.axis_categories();
        if env.show_x_text {
            if !x_categories.is_empty() {
                for (i, label) in x_categories.iter().enumerate() {
                    text(
                        svg,
                        (area.x + (i as f32 + 0.5) * cell_w, bottom + 16.0),
                        label,
                        TextStyle::Category,
                        Anchor::Middle,
                        Color::Ink70.css(),
                        0.0,
                    );
                }
            } else {
                // Without a range the axis counts cells; cell i is centered on i.
                let (lo, hi) = self.x_range.unwrap_or((-0.5, nx as f32 - 0.5));
                let scale = Scale { lo, hi, px_lo: area.x, px_hi: right };
                let (ticks, decimals) = nice_ticks(lo, hi, 8);
                for tick in ticks.into_iter().filter(|t| self.x_range.is_some() || t.fract() == 0.0) {
                    line(svg, (scale.at(tick), bottom), (scale.at(tick), bottom + 4.0), AXIS, 1.0, "");
                    text(
                        svg,
                        (scale.at(tick), bottom + 16.0),
                        &tick_label(tick, decimals),
                        TextStyle::Tick,
                        Anchor::Middle,
                        Color::Ink48.css(),
                        0.0,
                    );
                }
            }
            if !self.x_label.is_empty() {
                text(
                    svg,
                    (area.x + area.w / 2.0, bottom + 36.0),
                    &self.x_label,
                    TextStyle::Label,
                    Anchor::Middle,
                    Color::Ink70.css(),
                    0.0,
                );
            }
        }
        let (y_px_first, y_px_last) = if self.y_grows_up() { (bottom, area.y) } else { (area.y, bottom) };
        if !y_categories.is_empty() {
            for (i, label) in y_categories.iter().enumerate() {
                let y = y_px_first + (i as f32 + 0.5) / ny as f32 * (y_px_last - y_px_first);
                text(svg, (area.x - 8.0, y + 4.0), label, TextStyle::Category, Anchor::End, Color::Ink70.css(), 0.0);
            }
        } else {
            let (lo, hi) = self.y_range.unwrap_or((-0.5, ny as f32 - 0.5));
            let scale = Scale { lo, hi, px_lo: y_px_first, px_hi: y_px_last };
            let (ticks, decimals) = nice_ticks(lo, hi, (area.h / 40.0).round().max(2.0) as usize);
            for tick in ticks.into_iter().filter(|t| self.y_range.is_some() || t.fract() == 0.0) {
                line(svg, (area.x - 4.0, scale.at(tick)), (area.x, scale.at(tick)), AXIS, 1.0, "");
                text(
                    svg,
                    (area.x - 8.0, scale.at(tick) + 3.5),
                    &tick_label(tick, decimals),
                    TextStyle::Tick,
                    Anchor::End,
                    Color::Ink48.css(),
                    0.0,
                );
            }
        }
        if !self.y_label.is_empty() {
            text(svg, (20.0, area.y + area.h / 2.0), &self.y_label, TextStyle::Label, Anchor::Middle, Color::Ink70.css(), -90.0);
        }

        // Color bar: the ramp as a vertical strip with a few values beside it.
        if let Some(label) = &self.colorbar {
            let (bar_x, bar_w) = (right + 14.0, 10.0);
            let _ = write!(svg, "<linearGradient id=\"ramp{}\" x1=\"0\" y1=\"1\" x2=\"0\" y2=\"0\">", env.id);
            for stop in [0.0, 0.45, 0.85, 1.0] {
                let _ = write!(svg, "<stop offset=\"{stop}\" stop-color=\"{}\"/>", ramp_css(stop));
            }
            svg.push_str("</linearGradient>\n");
            let _ = writeln!(
                svg,
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"url(#ramp{})\" stroke=\"{AXIS}\" stroke-width=\"1\"/>",
                px(bar_x),
                px(area.y),
                px(bar_w),
                px(area.h),
                env.id
            );
            let scale = Scale { lo, hi, px_lo: bottom, px_hi: area.y };
            let (ticks, decimals) = nice_ticks(lo, hi, (area.h / 50.0).round().max(2.0) as usize);
            for tick in ticks {
                line(svg, (bar_x + bar_w, scale.at(tick)), (bar_x + bar_w + 3.0, scale.at(tick)), AXIS, 1.0, "");
                text(
                    svg,
                    (bar_x + bar_w + 6.0, scale.at(tick) + 3.5),
                    &tick_label(tick, decimals),
                    TextStyle::Tick,
                    Anchor::Start,
                    Color::Ink48.css(),
                    0.0,
                );
            }
            if !label.is_empty() {
                // Not an overline: units like "dB" must keep their capitalization.
                text(svg, (bar_x, area.y - 9.0), label, TextStyle::Tick, Anchor::Start, Color::Ink48.css(), 0.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bar charts
// ---------------------------------------------------------------------------

/// A bar per category: class probabilities, accuracy per word.
#[derive(Clone, Debug)]
pub struct BarChart {
    title: String,
    labels: Vec<String>,
    values: Vec<f32>,
    colors: Vec<Color>,
    y_label: String,
    x_label: String,
    y_range: Option<(f32, f32)>,
    value_decimals: Option<usize>,
    height: f32,
}

impl BarChart {
    /// One bar per `(label, value)` pair, all in a neutral gray. Use
    /// [`BarChart::highlight`] to pick out the ones that matter.
    pub fn new(title: &str, labels: &[&str], values: &[f32]) -> Self {
        assert_eq!(labels.len(), values.len(), "{} labels but {} values", labels.len(), values.len());
        BarChart {
            title: title.to_string(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            values: values.to_vec(),
            colors: vec![Color::Ink30; values.len()],
            y_label: String::new(),
            x_label: String::new(),
            y_range: None,
            value_decimals: None,
            height: 200.0,
        }
    }

    /// Color bar `index` (green for the right answer, red for a wrong one).
    pub fn highlight(mut self, index: usize, color: Color) -> Self {
        self.colors[index] = color;
        self
    }

    /// Give every bar the same color.
    pub fn color(mut self, color: Color) -> Self {
        self.colors.fill(color);
        self
    }

    /// Label beside the y axis.
    pub fn y_label(mut self, label: &str) -> Self {
        self.y_label = label.to_string();
        self
    }

    /// Label under the category names.
    pub fn x_label(mut self, label: &str) -> Self {
        self.x_label = label.to_string();
        self
    }

    /// Fix the y axis to `lo..hi` (e.g. `0.0..1.0` for probabilities).
    pub fn y_range(mut self, lo: f32, hi: f32) -> Self {
        self.y_range = Some((lo, hi));
        self
    }

    /// Print each bar's value above it, with this many decimals.
    pub fn show_values(mut self, decimals: usize) -> Self {
        self.value_decimals = Some(decimals);
        self
    }

    /// Height of the plot area in pixels (default 200).
    pub fn height(mut self, height: f32) -> Self {
        self.height = height;
        self
    }

    /// Render as a standalone SVG document.
    pub fn to_svg(&self) -> String {
        Figure::new(&self.title).panel(self.clone()).to_svg()
    }

    /// Render and write to `path`.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        Figure::new(&self.title).panel(self.clone()).save(path)
    }

    fn header_height(&self, show_title: bool) -> f32 {
        if show_title && !self.title.is_empty() {
            22.0
        } else {
            4.0
        }
    }

    /// Category names that do not fit side by side are turned 45 degrees.
    fn labels_are_rotated(&self, plot_w: f32) -> bool {
        let slot = plot_w / self.labels.len().max(1) as f32;
        self.labels.iter().any(|l| text_width(l, TextStyle::Category) > slot - 6.0)
    }

    fn footer_height(&self, plot_w: f32, show_x_text: bool) -> f32 {
        if !show_x_text {
            return 10.0;
        }
        let longest = self.labels.iter().map(|l| text_width(l, TextStyle::Category)).fold(0.0, f32::max);
        let labels = if self.labels_are_rotated(plot_w) { 12.0 + longest * 0.71 } else { 22.0 };
        labels + if self.x_label.is_empty() { 4.0 } else { 22.0 }
    }

    fn range(&self) -> (f32, f32) {
        self.y_range.unwrap_or_else(|| {
            let hi = self.values.iter().copied().filter(|v| v.is_finite()).fold(0.0, f32::max);
            let lo = self.values.iter().copied().filter(|v| v.is_finite()).fold(0.0, f32::min);
            if hi == lo {
                (0.0, 1.0)
            } else {
                (lo * 1.08, hi * 1.08)
            }
        })
    }

    fn draw(&self, svg: &mut String, area: Area, env: &Env) {
        let (lo, hi) = self.range();
        let ys = Scale { lo, hi, px_lo: area.y + area.h, px_hi: area.y };
        let (bottom, right) = (area.y + area.h, area.x + area.w);

        if env.show_title && !self.title.is_empty() {
            text(svg, (area.x, area.y - 9.0), &self.title, TextStyle::Overline, Anchor::Start, Color::Ink48.css(), 0.0);
        }
        let (ticks, decimals) = nice_ticks(lo, hi, (area.h / 45.0).round().max(2.0) as usize);
        for &tick in &ticks {
            line(svg, (area.x, ys.at(tick)), (right, ys.at(tick)), HAIRLINE, 1.0, "");
            text(
                svg,
                (area.x - 8.0, ys.at(tick) + 3.5),
                &tick_label(tick, decimals),
                TextStyle::Tick,
                Anchor::End,
                Color::Ink48.css(),
                0.0,
            );
        }

        let slot = area.w / self.values.len().max(1) as f32;
        let bar_w = (slot * 0.7).min(56.0);
        let baseline = ys.at(0.0f32.clamp(lo, hi));
        let rotated = self.labels_are_rotated(area.w);
        for (i, (&value, label)) in self.values.iter().zip(&self.labels).enumerate() {
            let center = area.x + (i as f32 + 0.5) * slot;
            let top = ys.at(value.clamp(lo, hi));
            let _ = writeln!(
                svg,
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" rx=\"1.5\" fill=\"{}\"/>",
                px(center - bar_w / 2.0),
                px(top.min(baseline)),
                px(bar_w),
                px((baseline - top).abs()),
                self.colors[i].solid()
            );
            if let Some(decimals) = self.value_decimals {
                text(
                    svg,
                    (center, top.min(baseline) - 5.0),
                    &tick_label(value, decimals),
                    TextStyle::Tick,
                    Anchor::Middle,
                    Color::Ink70.css(),
                    0.0,
                );
            }
            if env.show_x_text {
                // A highlighted bar gets a full-strength label.
                let fill = if self.colors[i] == Color::Ink30 { Color::Ink70.css() } else { INK };
                if rotated {
                    text(svg, (center + 3.0, bottom + 12.0), label, TextStyle::Category, Anchor::End, fill, -45.0);
                } else {
                    text(svg, (center, bottom + 16.0), label, TextStyle::Category, Anchor::Middle, fill, 0.0);
                }
            }
        }
        line(svg, (area.x, area.y), (area.x, bottom), AXIS, 1.0, "");
        line(svg, (area.x, baseline), (right, baseline), AXIS, 1.0, "");

        if env.show_x_text && !self.x_label.is_empty() {
            let y = bottom + self.footer_height(area.w, true) - 8.0;
            text(svg, (area.x + area.w / 2.0, y), &self.x_label, TextStyle::Label, Anchor::Middle, Color::Ink70.css(), 0.0);
        }
        if !self.y_label.is_empty() {
            text(svg, (20.0, area.y + area.h / 2.0), &self.y_label, TextStyle::Label, Anchor::Middle, Color::Ink70.css(), -90.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Figures: one or more panels on a sheet of paper
// ---------------------------------------------------------------------------

/// One chart inside a [`Figure`]. You rarely name this type: `Figure::panel`
/// accepts a [`LinePlot`], [`Heatmap`] or [`BarChart`] directly.
#[derive(Clone, Debug)]
pub enum Panel {
    /// A line plot.
    Line(LinePlot),
    /// A heatmap.
    Heat(Heatmap),
    /// A bar chart.
    Bar(BarChart),
}

impl From<LinePlot> for Panel {
    fn from(plot: LinePlot) -> Panel {
        Panel::Line(plot)
    }
}

impl From<Heatmap> for Panel {
    fn from(heatmap: Heatmap) -> Panel {
        Panel::Heat(heatmap)
    }
}

impl From<BarChart> for Panel {
    fn from(chart: BarChart) -> Panel {
        Panel::Bar(chart)
    }
}

/// What a panel needs to know about its surroundings while drawing.
struct Env {
    /// A number unique within the figure, for `id` attributes.
    id: usize,
    /// Whether to print the panel's own title above it.
    show_title: bool,
    /// Whether to print x tick labels and the x-axis label (off for the
    /// upper panels of a shared-axis stack).
    show_x_text: bool,
}

impl Panel {
    fn title(&self) -> &str {
        match self {
            Panel::Line(p) => &p.title,
            Panel::Heat(p) => &p.title,
            Panel::Bar(p) => &p.title,
        }
    }

    fn axis_labels(&self) -> (&str, &str) {
        match self {
            Panel::Line(p) => (&p.x_label, &p.y_label),
            Panel::Heat(p) => (&p.x_label, &p.y_label),
            Panel::Bar(p) => (&p.x_label, &p.y_label),
        }
    }

    fn has_colorbar(&self) -> bool {
        matches!(self, Panel::Heat(h) if h.colorbar.is_some())
    }

    /// (header height, plot width, plot height, footer height) in pixels.
    fn layout(&self, available_w: f32, env: &Env) -> (f32, f32, f32, f32) {
        let text_footer = |x_label: &str| match (env.show_x_text, x_label.is_empty()) {
            (false, _) => 10.0,
            (true, true) => 24.0,
            (true, false) => 44.0,
        };
        match self {
            Panel::Line(p) => (p.header_height(env.show_title), available_w, p.height, text_footer(&p.x_label)),
            Panel::Heat(p) => {
                let (w, h) = p.drawn_size(available_w);
                (p.header_height(env.show_title), w, h, text_footer(&p.x_label))
            }
            Panel::Bar(p) => {
                (p.header_height(env.show_title), available_w, p.height, p.footer_height(available_w, env.show_x_text))
            }
        }
    }

    fn draw(&self, svg: &mut String, area: Area, env: &Env) {
        match self {
            Panel::Line(p) => p.draw(svg, area, env),
            Panel::Heat(p) => p.draw(svg, area, env),
            Panel::Bar(p) => p.draw(svg, area, env),
        }
    }
}

/// A complete figure: a title and one or more panels stacked vertically, on
/// the paper background.
#[derive(Clone, Debug)]
pub struct Figure {
    title: String,
    desc: String,
    panels: Vec<Panel>,
    share_x: bool,
}

impl Figure {
    /// An empty figure with a title (shown top left, and used as the SVG's
    /// accessible `<title>`).
    pub fn new(title: &str) -> Self {
        Figure { title: title.to_string(), desc: String::new(), panels: Vec::new(), share_x: false }
    }

    /// A sentence describing the figure for readers who cannot see it (the
    /// SVG `<desc>`). If you do not give one, a plain description is put
    /// together from the panel titles and axis labels.
    pub fn desc(mut self, desc: &str) -> Self {
        self.desc = desc.to_string();
        self
    }

    /// Add a panel below the ones already there.
    pub fn panel(mut self, panel: impl Into<Panel>) -> Self {
        self.panels.push(panel.into());
        self
    }

    /// Declare that all panels have the same x axis: only the bottom panel
    /// then shows x tick labels and the x-axis label, and the panels move
    /// closer together. (The plot areas always line up; give the panels the
    /// same `x_range` and the data lines up too.)
    pub fn share_x(mut self) -> Self {
        self.share_x = true;
        self
    }

    fn auto_desc(&self) -> String {
        let mut parts = Vec::new();
        for panel in &self.panels {
            let kind = match panel {
                Panel::Line(_) => "Line plot",
                Panel::Heat(_) => "Heatmap",
                Panel::Bar(_) => "Bar chart",
            };
            let (x, y) = panel.axis_labels();
            let mut part = kind.to_string();
            if !panel.title().is_empty() && panel.title() != self.title {
                part.push_str(&format!(" \"{}\"", panel.title()));
            }
            if !x.is_empty() || !y.is_empty() {
                part.push_str(&format!(
                    " ({})",
                    [y, x].iter().filter(|s| !s.is_empty()).copied().collect::<Vec<_>>().join(" against ")
                ));
            }
            if let Panel::Line(p) = panel {
                let names: Vec<&str> = p.series.iter().map(|s| s.label.as_str()).filter(|l| !l.is_empty()).collect();
                if !names.is_empty() {
                    part.push_str(&format!(": {}", names.join(", ")));
                }
            }
            parts.push(part);
        }
        format!("{}.", parts.join("; "))
    }

    /// Render the figure as an SVG document.
    pub fn to_svg(&self) -> String {
        let right_margin = if self.panels.iter().any(Panel::has_colorbar) { MARGIN_RIGHT_COLORBAR } else { MARGIN_RIGHT };
        let available_w = WIDTH - MARGIN_LEFT - right_margin;
        // A single panel uses the figure title as its own; in a stack every
        // panel gets its title as a small label.
        let show_titles = self.panels.len() > 1;

        let mut body = String::new();
        let mut y = 14.0;
        if !self.title.is_empty() {
            text(&mut body, (24.0, y + 16.0), &self.title, TextStyle::Title, Anchor::Start, INK, 0.0);
            y += 32.0;
        }
        for (i, panel) in self.panels.iter().enumerate() {
            let is_last = i + 1 == self.panels.len();
            let env = Env { id: i, show_title: show_titles, show_x_text: is_last || !self.share_x };
            let (header, plot_w, plot_h, footer) = panel.layout(available_w, &env);
            y += header;
            panel.draw(&mut body, Area { x: MARGIN_LEFT, y, w: plot_w, h: plot_h }, &env);
            y += plot_h + footer + if is_last { 0.0 } else { 6.0 };
        }
        let height = (y + 8.0).ceil();

        let desc = if self.desc.is_empty() { self.auto_desc() } else { self.desc.clone() };
        let mut svg = String::with_capacity(body.len() + 512);
        let _ = writeln!(
            svg,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\" role=\"img\">",
            w = WIDTH,
            h = height
        );
        let _ = writeln!(svg, "<rect width=\"100%\" height=\"100%\" fill=\"{PAPER}\"/>");
        let _ = writeln!(svg, "<title>{}</title>", escape_xml(&self.title));
        let _ = writeln!(svg, "<desc>{}</desc>", escape_xml(&desc));
        svg.push_str(&body);
        svg.push_str("</svg>\n");
        svg
    }

    /// Render and write to `path`, creating the directory if needed.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, self.to_svg())
    }
}

// ---------------------------------------------------------------------------
// A minimal PNG writer, for embedding big heatmaps
// ---------------------------------------------------------------------------

/// Encode an image with one byte per pixel, each byte an index into
/// `palette`, as a PNG file.
///
/// PNG normally compresses its pixel data with DEFLATE. Writing a compressor
/// is a chapter of its own, but DEFLATE also allows "stored" blocks that are
/// not compressed at all, and that is what we use. A spectrogram comes to a
/// few tens of kilobytes, which is fine.
fn encode_png(width: usize, height: usize, pixels: &[u8], palette: &[[u8; 3]]) -> Vec<u8> {
    assert_eq!(pixels.len(), width * height, "pixel buffer does not match {width}x{height}");

    // Every row of pixels is preceded by a "filter type" byte; 0 = none.
    let mut raw = Vec::with_capacity((width + 1) * height);
    for row in pixels.chunks_exact(width.max(1)) {
        raw.push(0);
        raw.extend_from_slice(row);
    }

    // The zlib wrapper: 2 header bytes, stored blocks of at most 65535 bytes
    // (each with its length and the length's complement), an Adler-32 checksum.
    let mut zlib = vec![0x78, 0x01];
    let mut blocks = raw.chunks(65535).peekable();
    if blocks.peek().is_none() {
        zlib.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]); // one empty final block
    }
    while let Some(block) = blocks.next() {
        zlib.push(if blocks.peek().is_none() { 1 } else { 0 }); // 1 marks the final block
        zlib.extend_from_slice(&(block.len() as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut header = Vec::new();
    header.extend_from_slice(&(width as u32).to_be_bytes());
    header.extend_from_slice(&(height as u32).to_be_bytes());
    header.extend_from_slice(&[8, 3, 0, 0, 0]); // 8 bits per pixel, palette color, no interlacing

    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    png_chunk(&mut png, b"IHDR", &header);
    png_chunk(&mut png, b"PLTE", &palette.concat());
    png_chunk(&mut png, b"IDAT", &zlib);
    png_chunk(&mut png, b"IEND", &[]);
    png
}

/// Append one PNG chunk: length, 4-letter type, data, CRC of type and data.
fn png_chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend_from_slice(kind);
    png.extend_from_slice(data);
    let crc = crc32(&png[start..]);
    png.extend_from_slice(&crc.to_be_bytes());
}

/// CRC-32 as used by PNG, zip and Ethernet, one bit at a time.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// Adler-32, zlib's checksum: two running sums modulo 65521.
fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Base64: three bytes become four printable characters.
fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { ALPHABET[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[n as usize & 63] as char } else { '=' });
    }
    out
}

// ---------------------------------------------------------------------------
// Terminal renderers
// ---------------------------------------------------------------------------

/// Draw a waveform with block characters: `width` columns by `height` rows.
///
/// Each column covers an equal slice of the samples and is filled from the
/// lowest to the highest sample in that slice, so what you see is the
/// outline (the envelope) of the wave. The vertical scale is fixed: the top
/// edge is +1.0 and the bottom edge is -1.0, unless the signal is very
/// quiet, in which case it is scaled to its own peak. Half-height blocks
/// (`▀`, `▄`) double the vertical resolution.
pub fn waveform_ascii(samples: &[f32], width: usize, height: usize) -> String {
    if samples.is_empty() || width == 0 || height == 0 {
        return String::new();
    }
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let full_scale = if peak > 0.0 && peak < 0.5 { peak } else { 1.0 };
    let levels = 2 * height; // two half-rows per text row
                             // Map a sample value to a half-row index: 0 is the top (+full_scale).
    let level_of = |v: f32| (((1.0 - v / full_scale) / 2.0 * levels as f32) as usize).min(levels - 1);

    // Per column: the half-rows from its highest to its lowest sample.
    let mut columns = Vec::with_capacity(width);
    for col in 0..width {
        let from = col * samples.len() / width;
        let to = ((col + 1) * samples.len() / width).max(from + 1).min(samples.len());
        let slice = &samples[from.min(samples.len() - 1)..to];
        let (lo, hi) = slice.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &s| (lo.min(s), hi.max(s)));
        // A stretch quieter than half a text row is drawn as a thin line
        // through its middle, rather than as a fat bar straddling zero.
        let is_flat = (hi - lo) / full_scale < 2.0 / levels as f32;
        // (Nudged down a quarter of a half-row, so that silence, which
        // hovers around zero, does not flicker between two half-rows.)
        let middle = level_of((lo + hi) / 2.0 - full_scale / (2 * levels) as f32);
        columns.push(if is_flat { (middle, middle) } else { (level_of(hi), level_of(lo)) });
    }

    let mut out = String::with_capacity((width * 3 + 1) * height);
    for row in 0..height {
        for &(top, bottom) in &columns {
            let upper = (top..=bottom).contains(&(2 * row));
            let lower = (top..=bottom).contains(&(2 * row + 1));
            out.push(match (upper, lower) {
                (true, true) => '█',
                (true, false) => '▀',
                (false, true) => '▄',
                (false, false) => ' ',
            });
        }
        // Trailing blanks are noise in a terminal.
        let trimmed = out.trim_end_matches(' ').len();
        out.truncate(trimmed);
        out.push('\n');
    }
    out
}

/// Draw a matrix in the terminal with shade characters (` ░▒▓█`), resampled
/// to `width` columns by `height` rows. Each character shows the largest
/// value of the cells it covers, on a scale from the matrix's minimum to its
/// maximum. `orientation` works as for [`Heatmap`]: use
/// [`Orientation::TimeOnX`] for spectrograms and feature matrices.
pub fn heatmap_ascii(values: &Mat, width: usize, height: usize, orientation: Orientation) -> String {
    const SHADES: [char; 5] = [' ', '░', '▒', '▓', '█'];
    if values.rows == 0 || values.cols == 0 || width == 0 || height == 0 {
        return String::new();
    }
    let finite = values.data.iter().copied().filter(|v| v.is_finite());
    let (lo, hi) = finite.fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)));
    let span = if hi > lo { hi - lo } else { 1.0 };

    // (cells across, cells down) as displayed, and how to look one up.
    let (nx, ny) = match orientation {
        Orientation::TimeOnX => (values.rows, values.cols),
        _ => (values.cols, values.rows),
    };
    let value_at = |gx: usize, gy: usize| match orientation {
        Orientation::Matrix => values[(gy, gx)],
        Orientation::MatrixFlipped => values[(values.rows - 1 - gy, gx)],
        Orientation::TimeOnX => values[(gx, values.cols - 1 - gy)],
    };
    let (width, height) = (width.min(nx), height.min(ny));

    let mut out = String::with_capacity((width * 3 + 1) * height);
    for row in 0..height {
        let (y0, y1) = (row * ny / height, ((row + 1) * ny / height).max(row * ny / height + 1));
        for col in 0..width {
            let (x0, x1) = (col * nx / width, ((col + 1) * nx / width).max(col * nx / width + 1));
            let mut largest = f32::NEG_INFINITY;
            for gy in y0..y1 {
                for gx in x0..x1 {
                    largest = largest.max(value_at(gx, gy));
                }
            }
            let t = ((largest - lo) / span).clamp(0.0, 1.0);
            out.push(SHADES[((t * 5.0) as usize).min(4)]);
        }
        out.push('\n');
    }
    out
}

/// One line of bars (`▁▂▃▄▅▆▇█`), one per value, scaled from the smallest
/// value to the largest. Handy for a loudness contour.
pub fn sparkline(values: &[f32]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let (lo, hi) = values.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let span = if hi > lo { hi - lo } else { 1.0 };
    values.iter().map(|&v| BARS[(((v - lo) / span * 8.0) as usize).min(7)]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny XML well-formedness check: every tag that opens is closed, in
    /// the right order, and no stray `<` or `&` appears in text.
    fn assert_well_formed(svg: &str) {
        let mut stack: Vec<String> = Vec::new();
        let mut rest = svg;
        while let Some(open) = rest.find('<') {
            let text_before = &rest[..open];
            assert_valid_text(text_before);
            let close = open + rest[open..].find('>').expect("unterminated tag");
            let tag = &rest[open + 1..close];
            assert!(!tag.contains('<'), "'<' inside a tag: {tag}");
            if let Some(name) = tag.strip_prefix('/') {
                assert_eq!(stack.pop().as_deref(), Some(name), "mismatched closing tag </{name}>");
            } else if !tag.ends_with('/') {
                stack.push(tag.split_whitespace().next().unwrap().to_string());
            }
            // Attribute values must be quoted and balanced.
            assert_eq!(tag.matches('"').count() % 2, 0, "unbalanced quotes in <{tag}>");
            rest = &rest[close + 1..];
        }
        assert!(stack.is_empty(), "unclosed tags: {stack:?}");
    }

    fn assert_valid_text(text: &str) {
        let mut rest = text;
        while let Some(amp) = rest.find('&') {
            let entity_end = rest[amp..].find(';').expect("bare '&' in text");
            let entity = &rest[amp + 1..amp + entity_end];
            assert!(["amp", "lt", "gt", "quot"].contains(&entity), "unknown entity &{entity};");
            rest = &rest[amp + entity_end + 1..];
        }
    }

    /// The text content of every `<text>` element.
    fn texts(svg: &str) -> Vec<String> {
        svg.lines()
            .filter(|l| l.starts_with("<text"))
            .map(|l| l[l.find('>').unwrap() + 1..l.rfind("</text>").unwrap()].to_string())
            .collect()
    }

    fn sample_line_plot() -> LinePlot {
        let xs: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
        let ys: Vec<f32> = xs.iter().map(|x| (x * std::f32::consts::TAU).sin()).collect();
        LinePlot::new("A <sine> & friends")
            .x_label("time (s)")
            .y_label("amplitude")
            .line("sin", &xs, &ys)
            .series(Series::new("flat", &xs, &vec![0.0; 101]).dashed())
    }

    #[test]
    fn nice_ticks_are_round_numbers_inside_the_range() {
        assert_eq!(nice_ticks(0.0, 1.0, 5), (vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 1));
        assert_eq!(nice_ticks(0.0, 8000.0, 4), (vec![0.0, 2000.0, 4000.0, 6000.0, 8000.0], 0));
        assert_eq!(nice_ticks(-1.0, 1.0, 4), (vec![-1.0, -0.5, 0.0, 0.5, 1.0], 1));
        let (ticks, decimals) = nice_ticks(0.013, 0.587, 6);
        assert_eq!(decimals, 1);
        assert_eq!(ticks.len(), 5); // 0.1 ..= 0.5
        assert!((ticks[0] - 0.1).abs() < 1e-6 && (ticks[4] - 0.5).abs() < 1e-6);
        assert_eq!(nice_ticks(3.0, 3.0, 5).0, vec![3.0]);
    }

    #[test]
    fn tick_labels_use_a_real_minus_and_never_print_negative_zero() {
        assert_eq!(tick_label(-0.5, 1), "\u{2212}0.5");
        assert_eq!(tick_label(-0.0001, 1), "0.0");
        assert_eq!(tick_label(4000.0, 0), "4000");
    }

    #[test]
    fn line_plot_is_well_formed_and_follows_the_house_rules() {
        let svg = sample_line_plot().to_svg();
        assert_well_formed(&svg);
        // Paper background is the first child; title and desc follow.
        let mut lines = svg.lines();
        assert!(lines.next().unwrap().starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\""));
        assert_eq!(lines.next().unwrap(), "<rect width=\"100%\" height=\"100%\" fill=\"#F5F3EF\"/>");
        assert_eq!(lines.next().unwrap(), "<title>A &lt;sine&gt; &amp; friends</title>");
        assert!(lines.next().unwrap().starts_with("<desc>Line plot (amplitude against time (s)): sin, flat.</desc>"));
        // Self-contained: nothing is fetched from elsewhere.
        assert!(!svg.contains("http://") || svg.matches("http://").count() == 1, "only the xmlns may be a URL");
        assert!(!svg.contains("<script") && !svg.contains("<style") && !svg.contains("@import"));
        assert!(svg.contains("#047857") && svg.contains("#B45309"), "first two series are green and amber");
    }

    #[test]
    fn line_plot_axes_cover_the_data() {
        let labels = texts(&sample_line_plot().to_svg());
        // x from 0 to 1 in steps of 0.1 or 0.2; y ticks include -1, 0 and 1.
        for expected in ["0.0", "1.0", "\u{2212}1.0", "0.5", "time (s)", "amplitude", "sin", "flat"] {
            assert!(labels.iter().any(|l| l == expected), "missing label {expected:?} in {labels:?}");
        }
        let ((x_lo, x_hi), (y_lo, y_hi)) = sample_line_plot().ranges();
        assert_eq!((x_lo, x_hi), (0.0, 1.0));
        assert!(y_lo < -1.0 && y_lo > -1.2 && y_hi > 1.0 && y_hi < 1.2);
        assert_eq!(sample_line_plot().y_range(-2.0, 2.0).ranges().1, (-2.0, 2.0));
    }

    #[test]
    fn data_is_mapped_to_the_right_pixels() {
        // Three points on a fixed 0..2 by 0..1 range: the polyline must run
        // from the bottom-left corner of the plot area to the top-right.
        let svg = LinePlot::new("").line("", &[0.0, 1.0, 2.0], &[0.0, 0.5, 1.0]).y_range(0.0, 1.0).height(100.0).to_svg();
        let polyline = svg.lines().find(|l| l.starts_with("<polyline")).unwrap();
        // No title: the plot area starts at y = 14 + 4 = 18 and is 100 high;
        // x runs from 68 to 776.
        assert!(polyline.contains("points=\"68,118 422,68 776,18\""), "{polyline}");
    }

    #[test]
    fn long_series_are_thinned_but_keep_their_extremes() {
        let ys: Vec<f32> = (0..50_000).map(|i| if i == 25_000 { 9.0 } else { (i as f32 * 0.01).sin() }).collect();
        let svg = LinePlot::new("long").line_y("", &ys).to_svg();
        assert_well_formed(&svg);
        assert!(svg.len() < 60_000, "SVG is {} bytes", svg.len());
        // The spike at 9.0 must survive: it defines the top of the y range,
        // so some point is drawn at the very top of the plot area.
        let points = min_max_per_pixel(&[(0.2, 5.0), (0.4, 1.0), (0.6, 9.0), (0.8, 4.0), (1.5, 2.0)]);
        assert_eq!(points, vec![(0.2, 5.0), (0.4, 1.0), (0.6, 9.0), (0.8, 4.0), (1.5, 2.0)]);
        let points = min_max_per_pixel(&[(0.1, 5.0), (0.2, 6.0), (0.3, 1.0), (0.4, 3.0), (0.5, 9.0), (0.6, 4.0)]);
        assert_eq!(points, vec![(0.1, 5.0), (0.3, 1.0), (0.5, 9.0), (0.6, 4.0)]);
    }

    #[test]
    fn spans_markers_and_dots_are_drawn() {
        let svg = LinePlot::new("x")
            .series(Series::from_y("samples", &[0.0, 1.0, 0.0]).dots())
            .span(0.5, 1.5, "speech")
            .vline(1.0, "peak")
            .hline(0.5, "threshold")
            .annotate(1.0, 1.0, "here")
            .to_svg();
        assert_well_formed(&svg);
        assert_eq!(svg.matches("<circle").count(), 3);
        let labels = texts(&svg);
        for expected in ["SPEECH", "PEAK", "THRESHOLD", "here"] {
            assert!(labels.iter().any(|l| l == expected), "missing {expected}");
        }
        assert!(svg.contains("rgba(4,120,87,.10)"));
    }

    #[test]
    fn small_heatmap_draws_one_rect_per_cell_with_labels_and_overlays() {
        let values = Mat::from_vec(2, 3, vec![0.0, 0.5, 1.0, 1.0, 0.5, 0.0]);
        let svg = Heatmap::new("lattice", &values)
            .row_labels(&["\u{2423}", "c"])
            .col_labels(&["t0", "t1", "t2"])
            .cell_text(1)
            .path(&[(0, 0), (1, 1), (1, 2)], Color::Amber)
            .outline(1, 2, Color::Red)
            .colorbar("p")
            .x_label("frame")
            .to_svg();
        assert_well_formed(&svg);
        // 6 cells; the extremes use the ends of the ramp.
        assert_eq!(svg.matches("fill=\"#F5F3EF\"/>").count(), 1 + 2, "background + two zero cells");
        assert_eq!(svg.matches("fill=\"#191615\"/>").count(), 2, "two cells at the maximum");
        let labels = texts(&svg);
        for expected in ["\u{2423}", "c", "t0", "t2", "0.5", "1.0", "frame", "p"] {
            assert!(labels.iter().any(|l| l == expected), "missing {expected:?} in {labels:?}");
        }
        assert!(svg.contains("stroke=\"#B45309\"") && svg.contains("stroke=\"#B42318\""));
    }

    #[test]
    fn heatmap_orientation_puts_cells_where_documented() {
        let values = Mat::from_vec(2, 3, vec![0.0; 6]);
        let h = Heatmap::new("", &values);
        assert_eq!(h.grid_size(), (3, 2));
        assert_eq!(h.grid_position(0, 0), (0, 0)); // top left
        assert_eq!(h.grid_position(1, 2), (2, 1)); // bottom right
        let h = h.orientation(Orientation::MatrixFlipped);
        assert_eq!(h.grid_position(0, 0), (0, 1)); // bottom left
        let h = h.orientation(Orientation::TimeOnX);
        assert_eq!(h.grid_size(), (2, 3)); // rows become x
        assert_eq!(h.grid_position(0, 0), (0, 2)); // first frame, lowest bin: bottom left
        assert_eq!(h.grid_position(1, 2), (1, 0)); // last frame, highest bin: top right
    }

    #[test]
    fn spectrogram_sized_heatmap_embeds_an_image_and_labels_its_axes() {
        let values = Mat::from_fn(98, 257, |r, c| (r + c) as f32);
        let svg = Heatmap::new("spec", &values)
            .orientation(Orientation::TimeOnX)
            .x_range(0.0, 1.0)
            .y_range(0.0, 8000.0)
            .x_label("time (s)")
            .y_label("frequency (Hz)")
            .to_svg();
        assert_well_formed(&svg);
        assert_eq!(svg.matches("<image").count(), 1);
        assert!(svg.contains("href=\"data:image/png;base64,iVBORw0KGgo"), "PNG signature in base64");
        assert!(svg.len() < 60_000, "SVG is {} bytes", svg.len());
        let labels = texts(&svg);
        for expected in ["0", "8000", "4000", "0.0", "1.0", "frequency (Hz)"] {
            assert!(labels.iter().any(|l| l == expected), "missing {expected:?} in {labels:?}");
        }
    }

    #[test]
    fn bar_chart_scales_bars_and_rotates_crowded_labels() {
        let svg = BarChart::new("probabilities", &["yes", "no"], &[0.75, 0.25])
            .y_range(0.0, 1.0)
            .highlight(0, Color::Green)
            .show_values(2)
            .height(100.0)
            .to_svg();
        assert_well_formed(&svg);
        // Plot area: y from 14 + 32 + 4 = 50 to 150. 0.75 -> a bar 75 px tall.
        assert!(svg.contains("height=\"75\" rx=\"1.5\" fill=\"#047857\""), "{svg}");
        assert!(svg.contains("height=\"25\" rx=\"1.5\" fill=\"#B3B1AE\""));
        assert!(texts(&svg).iter().any(|l| l == "0.75"));
        assert!(!svg.contains("rotate(-45"));

        let words: Vec<String> = (0..35).map(|i| format!("word{i}")).collect();
        let labels: Vec<&str> = words.iter().map(String::as_str).collect();
        let svg = BarChart::new("many", &labels, &[0.1; 35]).to_svg();
        assert_well_formed(&svg);
        assert_eq!(svg.matches("rotate(-45").count(), 35);
    }

    #[test]
    fn stacked_panels_share_margins_and_hide_inner_x_labels() {
        let top = LinePlot::new("waveform").x_label("time (s)").line_y("", &[0.0, 1.0, 0.0]).x_range(0.0, 2.0);
        let bottom = Heatmap::new("spectrogram", &Mat::from_vec(2, 2, vec![0.0, 1.0, 2.0, 3.0]))
            .orientation(Orientation::TimeOnX)
            .x_range(0.0, 2.0)
            .x_label("time (s)")
            .colorbar("dB");
        let svg = Figure::new("Stack").desc("A waveform above its spectrogram.").panel(top).panel(bottom).share_x().to_svg();
        assert_well_formed(&svg);
        assert!(svg.contains("<desc>A waveform above its spectrogram.</desc>"));
        let labels = texts(&svg);
        assert_eq!(labels.iter().filter(|l| *l == "time (s)").count(), 1, "x label only under the last panel");
        assert!(labels.contains(&"WAVEFORM".to_string()) && labels.contains(&"SPECTROGRAM".to_string()));
        // Both plot areas span x = 68 .. 716 (800 - 84 for the color bar).
        assert!(svg.contains("<line x1=\"68\" y1=\"288\" x2=\"716\" y2=\"288\""), "bottom axis of the line plot");
        assert!(svg.contains("width=\"648\" height=\"200\" fill=\"none\""), "frame of the heatmap");
    }

    #[test]
    fn png_building_blocks_match_published_values() {
        assert_eq!(crc32(b"IEND"), 0xAE42_6082); // the last 4 bytes of every PNG file
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");

        let png = encode_png(2, 1, &[0, 1], &[[0, 0, 0], [255, 255, 255]]);
        assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        assert_eq!(&png[12..16], b"IHDR");
        assert_eq!(&png[16..24], &[0, 0, 0, 2, 0, 0, 0, 1]); // 2 x 1
        assert_eq!(&png[png.len() - 8..png.len() - 4], b"IEND");
    }

    #[test]
    fn ramp_runs_from_paper_to_ink() {
        assert_eq!(ramp(0.0), [0xF5, 0xF3, 0xEF]);
        assert_eq!(ramp(0.45), [0x9F, 0xB8, 0xA9]);
        assert_eq!(ramp(0.85), [0x04, 0x54, 0x3E]);
        assert_eq!(ramp(1.0), [0x19, 0x16, 0x15]);
        assert_eq!(ramp(7.0), ramp(1.0));
        assert_eq!(ramp(f32::NAN), ramp(0.0));
    }

    #[test]
    fn waveform_ascii_draws_the_envelope() {
        // Loud first half, silent second half, 4 columns x 2 rows.
        let mut samples = vec![0.0f32; 400];
        for (i, s) in samples.iter_mut().enumerate().take(200) {
            *s = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let art = waveform_ascii(&samples, 4, 2);
        // Silence sits just below the center line: the top half of row 1.
        assert_eq!(art, "██\n██▀▀\n");

        // Faint noise around zero stays one steady thin line.
        let noise: Vec<f32> = (0..400).map(|i| if i % 3 == 0 { 0.001 } else { -0.001 }).chain([1.0, -1.0]).collect();
        assert_eq!(waveform_ascii(&noise, 8, 2), "       █\n▀▀▀▀▀▀▀█\n");
        assert_eq!(waveform_ascii(&[], 10, 2), "");
        // A quiet signal is scaled up to its own peak.
        let quiet: Vec<f32> = samples.iter().map(|s| s * 0.01).collect();
        assert_eq!(waveform_ascii(&quiet, 4, 2), art);
    }

    #[test]
    fn heatmap_ascii_shades_by_value_and_respects_orientation() {
        let values = Mat::from_vec(2, 3, vec![0.0, 0.5, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(heatmap_ascii(&values, 3, 2, Orientation::Matrix), " ▒█\n   \n");
        assert_eq!(heatmap_ascii(&values, 3, 2, Orientation::MatrixFlipped), "   \n ▒█\n");
        // TimeOnX: 2 columns (rows of the matrix), highest column index on top.
        assert_eq!(heatmap_ascii(&values, 2, 3, Orientation::TimeOnX), "█ \n▒ \n  \n");
        // Downsampling keeps the maximum of the cells each character covers.
        assert_eq!(heatmap_ascii(&values, 1, 1, Orientation::Matrix), "█\n");
    }

    #[test]
    fn sparkline_scales_to_its_own_range() {
        assert_eq!(sparkline(&[0.0, 0.5, 1.0]), "▁▅█");
        assert_eq!(sparkline(&[]), "");
    }
}
