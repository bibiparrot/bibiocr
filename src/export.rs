use pandoc::{InputFormat, InputKind, OutputFormat, OutputKind, PandocOption};
use pulldown_cmark::{Options, Parser, html};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::LazyLock,
};

static MARKDOWN_IMAGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(!\[[^\]]*\]\(\s*)(<[^>]+>|[^\s)]+)"#).expect("Markdown image regex must compile")
});
static HTML_IMAGE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)(<img\b[^>]*\bsrc\s*=\s*[\"'])([^\"']+)([\"'])"#)
        .expect("HTML image regex must compile")
});

#[derive(Clone, Copy)]
enum LinkKind {
    Markdown,
    Html,
}

/// Save Markdown with portable file links. Every referenced local image is
/// copied beside the Markdown file and its link is rewritten to that filename.
pub fn save_markdown(path: &Path, markdown: &str, search_dirs: &[PathBuf]) -> Result<(), String> {
    let destination_dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(destination_dir).map_err(markdown_error)?;
    let rewritten = copy_local_images(markdown, destination_dir, search_dirs)?;
    fs::write(path, rewritten).map_err(markdown_error)
}

/// Convert Markdown (including raw HTML blocks) to HTML, then let Pandoc write
/// the DOCX package. Resource paths allow Pandoc to resolve OCR image links.
pub fn save_docx(
    path: &Path,
    markdown: &str,
    search_dirs: &[PathBuf],
    pandoc_executable: &Path,
) -> Result<(), String> {
    if !pandoc_executable.is_file() {
        return Err(
            rust_i18n::t!("pandoc_missing", path = pandoc_executable.display()).into_owned(),
        );
    }
    let pandoc_directory = pandoc_executable.parent().ok_or_else(|| {
        rust_i18n::t!("pandoc_missing", path = pandoc_executable.display()).into_owned()
    })?;
    let mut resources = Vec::new();
    if let Some(output_directory) = path.parent() {
        resources.push(output_directory.to_path_buf());
    }
    for directory in search_dirs {
        if !resources.contains(directory) {
            resources.push(directory.clone());
        }
    }

    let mut converter = pandoc::new();
    converter
        .add_pandoc_path_hint(pandoc_directory)
        .set_input(InputKind::Pipe(markdown_to_html(markdown)))
        .set_input_format(InputFormat::Html, Vec::new())
        .set_output(OutputKind::File(path.to_path_buf()))
        .set_output_format(OutputFormat::Docx, Vec::new());
    if !resources.is_empty() {
        converter.add_option(PandocOption::ResourcePath(resources));
    }
    converter.execute().map(|_| ()).map_err(docx_error)
}

fn markdown_to_html(markdown: &str) -> String {
    let mut body = String::new();
    html::push_html(&mut body, Parser::new_ext(markdown, Options::all()));
    format!("<!doctype html><html><head><meta charset=\"utf-8\"></head><body>{body}</body></html>")
}

fn copy_local_images(
    markdown: &str,
    destination_dir: &Path,
    search_dirs: &[PathBuf],
) -> Result<String, String> {
    let mut copied = HashMap::new();
    let mut reserved = HashSet::new();
    let markdown = rewrite_image_links(
        markdown,
        &MARKDOWN_IMAGE,
        2,
        LinkKind::Markdown,
        destination_dir,
        search_dirs,
        &mut copied,
        &mut reserved,
    )?;
    rewrite_image_links(
        &markdown,
        &HTML_IMAGE,
        2,
        LinkKind::Html,
        destination_dir,
        search_dirs,
        &mut copied,
        &mut reserved,
    )
}

#[allow(clippy::too_many_arguments)]
fn rewrite_image_links(
    input: &str,
    regex: &Regex,
    source_group: usize,
    link_kind: LinkKind,
    destination_dir: &Path,
    search_dirs: &[PathBuf],
    copied: &mut HashMap<PathBuf, String>,
    reserved: &mut HashSet<String>,
) -> Result<String, String> {
    let mut output = String::with_capacity(input.len());
    let mut cursor = 0;
    for captures in regex.captures_iter(input) {
        let Some(source_match) = captures.get(source_group) else {
            continue;
        };
        output.push_str(&input[cursor..source_match.start()]);
        let source = source_match.as_str();
        match copy_image(source, destination_dir, search_dirs, copied, reserved)? {
            Some(filename) => match link_kind {
                LinkKind::Markdown if filename.chars().any(char::is_whitespace) => {
                    output.push('<');
                    output.push_str(&filename);
                    output.push('>');
                }
                LinkKind::Markdown => output.push_str(&filename),
                LinkKind::Html => output.push_str(&escape_html_attribute(&filename)),
            },
            None => output.push_str(source),
        }
        cursor = source_match.end();
    }
    output.push_str(&input[cursor..]);
    Ok(output)
}

fn copy_image(
    source: &str,
    destination_dir: &Path,
    search_dirs: &[PathBuf],
    copied: &mut HashMap<PathBuf, String>,
    reserved: &mut HashSet<String>,
) -> Result<Option<String>, String> {
    let source = source.trim().trim_matches(['<', '>']);
    let lower = source.to_ascii_lowercase();
    if lower.starts_with("data:")
        || lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with('#')
    {
        return Ok(None);
    }

    let candidates = image_path_candidates(source, search_dirs);
    let source_path = candidates
        .iter()
        .find(|candidate| candidate.is_file())
        .cloned()
        .or_else(|| candidates.first().cloned())
        .unwrap_or_else(|| PathBuf::from(source));
    if !source_path.is_file() {
        return Err(image_copy_error(&source_path, "file does not exist"));
    }
    let identity = source_path
        .canonicalize()
        .unwrap_or_else(|_| source_path.clone());
    if let Some(filename) = copied.get(&identity) {
        return Ok(Some(filename.clone()));
    }
    let original_name = source_path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| image_copy_error(&source_path, "image has no filename"))?;
    let filename = available_filename(original_name, &source_path, destination_dir, reserved)?;
    let target = destination_dir.join(&filename);
    if !same_file(&source_path, &target) {
        fs::copy(&source_path, &target)
            .map_err(|error| image_copy_error(&source_path, &error.to_string()))?;
    }
    reserved.insert(filename_key(&filename));
    copied.insert(identity, filename.clone());
    Ok(Some(filename))
}

fn available_filename(
    original: &str,
    source: &Path,
    destination_dir: &Path,
    reserved: &HashSet<String>,
) -> Result<String, String> {
    let original_path = Path::new(original);
    let stem = original_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("image");
    let extension = original_path.extension().and_then(|value| value.to_str());
    for index in 1..=10_000 {
        let filename = if index == 1 {
            original.to_owned()
        } else if let Some(extension) = extension {
            format!("{stem}-{index}.{extension}")
        } else {
            format!("{stem}-{index}")
        };
        if reserved.contains(&filename_key(&filename)) {
            continue;
        }
        let target = destination_dir.join(&filename);
        if !target.exists() || same_file(source, &target) || same_contents(source, &target)? {
            return Ok(filename);
        }
    }
    Err(image_copy_error(source, "too many filename collisions"))
}

fn same_file(left: &Path, right: &Path) -> bool {
    left.canonicalize()
        .ok()
        .zip(right.canonicalize().ok())
        .is_some_and(|(left, right)| {
            if cfg!(windows) {
                left.to_string_lossy()
                    .eq_ignore_ascii_case(&right.to_string_lossy())
            } else {
                left == right
            }
        })
}

fn same_contents(left: &Path, right: &Path) -> Result<bool, String> {
    let left_metadata =
        fs::metadata(left).map_err(|error| image_copy_error(left, &error.to_string()))?;
    let right_metadata =
        fs::metadata(right).map_err(|error| image_copy_error(right, &error.to_string()))?;
    if left_metadata.len() != right_metadata.len() {
        return Ok(false);
    }
    let left_bytes = fs::read(left).map_err(|error| image_copy_error(left, &error.to_string()))?;
    let right_bytes =
        fs::read(right).map_err(|error| image_copy_error(right, &error.to_string()))?;
    Ok(left_bytes == right_bytes)
}

fn image_path_candidates(source: &str, search_dirs: &[PathBuf]) -> Vec<PathBuf> {
    if let Some(file_path) = source.strip_prefix("file://") {
        let file_path = if cfg!(target_os = "windows") {
            file_path.trim_start_matches('/')
        } else {
            file_path
        };
        return vec![PathBuf::from(file_path)];
    }
    let path = PathBuf::from(source.replace('/', std::path::MAIN_SEPARATOR_STR));
    if path.is_absolute() {
        return vec![path];
    }
    search_dirs.iter().map(|base| base.join(&path)).collect()
}

fn filename_key(filename: &str) -> String {
    if cfg!(windows) {
        filename.to_ascii_lowercase()
    } else {
        filename.to_owned()
    }
}

fn escape_html_attribute(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn markdown_error(error: impl std::fmt::Display) -> String {
    rust_i18n::t!("markdown_save_error", error = error).into_owned()
}

fn image_copy_error(path: &Path, error: &str) -> String {
    rust_i18n::t!("image_copy_error", path = path.display(), error = error).into_owned()
}

fn docx_error(error: impl std::fmt::Display) -> String {
    rust_i18n::t!("docx_save_error", error = error).into_owned()
}

#[cfg(test)]
mod tests {
    use super::{copy_local_images, markdown_to_html, save_docx, save_markdown};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn temp_dir(label: &str) -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!("bibiocr-{label}-{stamp}"))
    }

    #[test]
    fn copies_markdown_and_html_images_beside_document() {
        let source_dir = temp_dir("copy-source");
        let destination_dir = temp_dir("copy-destination");
        fs::create_dir_all(source_dir.join("imgs")).unwrap();
        fs::create_dir_all(&destination_dir).unwrap();
        fs::write(source_dir.join("imgs/page.png"), b"png bytes").unwrap();
        let source = "![page](imgs/page.png)\n<img src=\"imgs/page.png\">";
        let rewritten =
            copy_local_images(source, &destination_dir, std::slice::from_ref(&source_dir)).unwrap();
        assert_eq!(rewritten, "![page](page.png)\n<img src=\"page.png\">");
        assert_eq!(
            fs::read(destination_dir.join("page.png")).unwrap(),
            b"png bytes"
        );
        assert!(!rewritten.contains("base64"));
        let _ = fs::remove_dir_all(source_dir);
        let _ = fs::remove_dir_all(destination_dir);
    }

    #[test]
    fn saves_markdown_and_copies_its_images() {
        let source_dir = temp_dir("save-source");
        let destination_dir = temp_dir("save-destination");
        fs::create_dir_all(&source_dir).unwrap();
        fs::write(source_dir.join("scan.png"), b"scan").unwrap();
        let path = destination_dir.join("document.md");
        save_markdown(
            &path,
            "![scan](scan.png)",
            std::slice::from_ref(&source_dir),
        )
        .unwrap();
        assert_eq!(fs::read_to_string(path).unwrap(), "![scan](scan.png)");
        assert_eq!(fs::read(destination_dir.join("scan.png")).unwrap(), b"scan");
        let _ = fs::remove_dir_all(source_dir);
        let _ = fs::remove_dir_all(destination_dir);
    }

    #[test]
    fn markdown_html_keeps_raw_html_tables() {
        let html = markdown_to_html("# Title\n\n<table><tr><td>Cell</td></tr></table>");
        assert!(html.contains("<h1>Title</h1>"));
        assert!(html.contains("<table><tr><td>Cell</td></tr></table>"));
    }

    #[test]
    fn writes_docx_with_configured_pandoc_when_available() {
        let pandoc_path = crate::backend::pandoc_path().ok().or_else(|| {
            let repository_parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent()?;
            let candidate = repository_parent
                .join("pandoc-3.10.2-windows-x86_64")
                .join("pandoc-3.10.2")
                .join(if cfg!(windows) {
                    "pandoc.exe"
                } else {
                    "pandoc"
                });
            candidate.is_file().then_some(candidate)
        });
        let Some(pandoc_path) = pandoc_path else {
            return;
        };
        let directory = temp_dir("pandoc-docx");
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("output.docx");
        save_docx(
            &path,
            "# Title\n\n<table><tr><th>A</th></tr><tr><td>B</td></tr></table>",
            &[],
            &pandoc_path,
        )
        .expect("Pandoc HTML to DOCX export should succeed");
        let bytes = fs::read(&path).unwrap();
        assert!(bytes.starts_with(b"PK"));
        let _ = fs::remove_dir_all(directory);
    }
}
