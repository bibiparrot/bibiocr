use eframe::egui::{self, Color32, RichText, Stroke};
use std::hash::{Hash, Hasher};

/// Render raw HTML blocks passed through by `egui_commonmark`.
///
/// OCR output primarily contains HTML tables. This renderer also handles the
/// common block/line tags used in Markdown documents and safely displays the
/// text of unknown tags instead of dropping the whole block.
pub fn render_html(ui: &mut egui::Ui, html: &str) {
    let tables = extract_tables(html);
    if tables.is_empty() {
        render_text_block(ui, html);
        return;
    }

    let mut cursor = 0;
    for table in tables {
        if table.start > cursor {
            render_text_block(ui, &html[cursor..table.start]);
        }
        render_table(ui, &table.rows, &html[table.start..table.end]);
        cursor = table.end;
    }
    if cursor < html.len() {
        render_text_block(ui, &html[cursor..]);
    }
}

struct HtmlTable {
    start: usize,
    end: usize,
    rows: Vec<Vec<String>>,
}

fn render_text_block(ui: &mut egui::Ui, html: &str) {
    let text = html_to_text(html);
    if !text.trim().is_empty() {
        ui.add(egui::Label::new(text.trim()).wrap());
    }
}

fn render_table(ui: &mut egui::Ui, rows: &[Vec<String>], identity: &str) {
    if rows.is_empty() {
        return;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    identity.hash(&mut hasher);
    let grid_id = ("markdown-html-table", hasher.finish());
    let column_count = rows.iter().map(Vec::len).max().unwrap_or(1).max(1);
    egui::Frame::new()
        .stroke(Stroke::new(1.0, Color32::from_gray(190)))
        .inner_margin(6.0)
        .show(ui, |ui| {
            let spacing = ui.spacing().item_spacing.x;
            let cell_width = ((ui.available_width() - spacing * (column_count - 1) as f32)
                / column_count as f32)
                .max(48.0);
            egui::Grid::new(grid_id)
                .num_columns(column_count)
                .min_col_width(cell_width)
                .max_col_width(cell_width)
                .striped(true)
                .show(ui, |ui| {
                    for (row_index, row) in rows.iter().enumerate() {
                        for column_index in 0..column_count {
                            let cell = row.get(column_index).map(String::as_str).unwrap_or("");
                            let text = RichText::new(cell).strong_if(row_index == 0);
                            ui.add_sized([cell_width, 0.0], egui::Label::new(text).wrap());
                        }
                        ui.end_row();
                    }
                });
        });
}

trait RichTextExt {
    fn strong_if(self, condition: bool) -> Self;
}

impl RichTextExt for RichText {
    fn strong_if(self, condition: bool) -> Self {
        if condition { self.strong() } else { self }
    }
}

fn extract_tables(html: &str) -> Vec<HtmlTable> {
    let lower = html.to_ascii_lowercase();
    let mut result = Vec::new();
    let mut cursor = 0;
    while let Some(relative_start) = lower[cursor..].find("<table") {
        let start = cursor + relative_start;
        let Some(open_end) = lower[start..].find('>').map(|index| start + index + 1) else {
            break;
        };
        let Some(relative_end) = lower[open_end..].find("</table>") else {
            break;
        };
        let end = open_end + relative_end + "</table>".len();
        result.push(HtmlTable {
            start,
            end,
            rows: extract_rows(&html[open_end..open_end + relative_end]),
        });
        cursor = end;
    }
    result
}

fn extract_rows(table_body: &str) -> Vec<Vec<String>> {
    let lower = table_body.to_ascii_lowercase();
    let mut rows = Vec::new();
    let mut cursor = 0;
    while let Some(relative_start) = lower[cursor..].find("<tr") {
        let start = cursor + relative_start;
        let Some(open_end) = lower[start..].find('>').map(|index| start + index + 1) else {
            break;
        };
        let Some(relative_end) = lower[open_end..].find("</tr>") else {
            break;
        };
        let end = open_end + relative_end;
        let cells = extract_cells(&table_body[open_end..end]);
        if !cells.is_empty() {
            rows.push(cells);
        }
        cursor = end + "</tr>".len();
    }
    rows
}

fn extract_cells(row_body: &str) -> Vec<String> {
    let lower = row_body.to_ascii_lowercase();
    let mut cells = Vec::new();
    let mut cursor = 0;
    while cursor < row_body.len() {
        let td = lower[cursor..].find("<td").map(|index| (index, "</td>"));
        let th = lower[cursor..].find("<th").map(|index| (index, "</th>"));
        let Some((relative_start, close_tag)) = [td, th].into_iter().flatten().min_by_key(|v| v.0)
        else {
            break;
        };
        let start = cursor + relative_start;
        let Some(open_end) = lower[start..].find('>').map(|index| start + index + 1) else {
            break;
        };
        let Some(relative_end) = lower[open_end..].find(close_tag) else {
            break;
        };
        let end = open_end + relative_end;
        cells.push(html_to_text(&row_body[open_end..end]).trim().to_owned());
        cursor = end + close_tag.len();
    }
    cells
}

pub(crate) fn html_to_text(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let bytes = html.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'<' {
            let Some(relative_end) = html[index..].find('>') else {
                text.push_str(&html[index..]);
                break;
            };
            let end = index + relative_end;
            let tag = html[index + 1..end]
                .trim()
                .trim_start_matches('/')
                .split_ascii_whitespace()
                .next()
                .unwrap_or("")
                .trim_end_matches('/')
                .to_ascii_lowercase();
            if matches!(
                tag.as_str(),
                "br" | "p" | "div" | "li" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
            ) && !text.ends_with('\n')
            {
                text.push('\n');
            }
            index = end + 1;
        } else if bytes[index] == b'&' {
            if let Some(relative_end) = html[index..].find(';') {
                let end = index + relative_end;
                if let Some(decoded) = decode_entity(&html[index + 1..end]) {
                    text.push(decoded);
                    index = end + 1;
                    continue;
                }
            }
            text.push('&');
            index += 1;
        } else {
            let ch = html[index..].chars().next().expect("valid UTF-8 boundary");
            text.push(ch);
            index += ch.len_utf8();
        }
    }
    text
}

fn decode_entity(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" | "#39" => Some('\''),
        "nbsp" => Some(' '),
        value if value.starts_with("#x") || value.starts_with("#X") => {
            u32::from_str_radix(&value[2..], 16)
                .ok()
                .and_then(char::from_u32)
        }
        value if value.starts_with('#') => value[1..].parse().ok().and_then(char::from_u32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{extract_tables, html_to_text, render_table};
    use eframe::egui;

    #[test]
    fn extracts_html_table_cells() {
        let tables = extract_tables(
            "before<table><tr><th>A</th><th>B</th></tr><tr><td>一</td><td>x&amp;y</td></tr></table>",
        );
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].rows[1], ["一", "x&y"]);
    }

    #[test]
    fn converts_common_html_to_text() {
        assert_eq!(
            html_to_text("<p>Hello<br>world &lt;3</p>").trim(),
            "Hello\nworld <3"
        );
    }

    #[test]
    fn table_uses_the_available_preview_width() {
        let context = egui::Context::default();
        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(800.0, 500.0),
            )),
            ..Default::default()
        };
        let mut measured = None;
        let mut output = context.run_ui(input.take(), |ui| {
            let available = ui.available_width();
            let response = ui.scope(|ui| {
                render_table(
                    ui,
                    &[
                        vec!["A".to_owned(), "B".to_owned()],
                        vec!["1".to_owned(), "2".to_owned()],
                    ],
                    "width-test",
                );
            });
            measured = Some((available, response.response.rect.width()));
        });
        output.textures_delta.clear();
        let (available, rendered) = measured.expect("table should render");
        assert!(
            rendered >= available * 0.95,
            "table width {rendered} should fill preview width {available}"
        );
    }
}
