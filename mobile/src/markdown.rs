pub fn normalize_ocr(source: &str) -> String {
    use pulldown_cmark::{Event, Parser};

    let mut output = String::with_capacity(source.len());
    let mut previous = 0;
    for (event, range) in Parser::new(source).into_offset_iter() {
        if matches!(event, Event::Html(_) | Event::InlineHtml(_)) {
            output.push_str(&source[previous..range.start]);
            let fragment = &source[range.clone()];
            output.push_str(&htmd::convert(fragment).unwrap_or_else(|_| fragment.to_owned()));
            previous = range.end;
        }
    }
    output.push_str(&source[previous..]);
    output
}

#[cfg(test)]
mod tests {
    #[test]
    fn mixed_ocr_markdown_has_no_raw_html_in_editor() {
        let source = "# 结果\n\n<div style=\"text-align: center;\"><img src=\"imgs/crop.jpg\" alt=\"Image\" width=\"7%\" /></div>\n\n正文 **重点**。";
        let output = super::normalize_ocr(source);
        assert!(output.starts_with("# 结果"), "{output}");
        assert!(output.contains("正文 **重点**。"), "{output}");
        assert!(!output.contains("<div"), "{output}");
        assert!(!output.contains("<img"), "{output}");
        assert!(output.contains("![Image]"), "{output}");
    }

    #[test]
    fn html_table_and_inline_tags_keep_their_text() {
        let source = "前文 <strong>重点</strong> 后文\n\n<table><tr><th>名称</th></tr><tr><td>金额</td></tr></table>";
        let output = super::normalize_ocr(source);
        assert!(output.contains("重点"), "{output}");
        assert!(output.contains("金额"), "{output}");
        assert!(!output.contains("<table"), "{output}");
        assert!(!output.contains("<strong"), "{output}");
    }
}
