#[derive(Clone, Debug, PartialEq)]
pub struct LayoutBlock {
    pub class_id: usize,
    pub label: &'static str,
    pub score: f32,
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub order: i32,
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub class_id: usize,
    pub score: f32,
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub model_order: i32,
}

const LABELS: [&str; 25] = [
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
];

fn iou(first: &Detection, second: &Detection) -> f32 {
    let width = (first.right.min(second.right) - first.left.max(second.left) + 1.0).max(0.0);
    let height = (first.bottom.min(second.bottom) - first.top.max(second.top) + 1.0).max(0.0);
    let intersection = width * height;
    let first_area = (first.right - first.left + 1.0) * (first.bottom - first.top + 1.0);
    let second_area = (second.right - second.left + 1.0) * (second.bottom - second.top + 1.0);
    let union = first_area + second_area - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

pub fn postprocess(mut detections: Vec<Detection>) -> Vec<LayoutBlock> {
    detections.retain(|item| {
        item.score > 0.3
            && item.class_id < LABELS.len()
            && item.right > item.left
            && item.bottom > item.top
    });
    detections.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut kept: Vec<Detection> = Vec::new();
    for detection in detections {
        let suppressed = kept.iter().any(|previous| {
            iou(&detection, previous)
                >= if detection.class_id == previous.class_id {
                    0.6
                } else {
                    0.98
                }
        });
        if !suppressed {
            kept.push(detection);
        }
    }
    kept.sort_by_key(|item| item.model_order);
    kept.into_iter()
        .enumerate()
        .map(|(index, detection)| LayoutBlock {
            class_id: detection.class_id,
            label: LABELS[detection.class_id],
            score: detection.score,
            left: detection.left,
            top: detection.top,
            right: detection.right,
            bottom: detection.bottom,
            order: index as i32 + 1,
        })
        .collect()
}

pub fn analyze(
    model_path: &std::path::Path,
    image_path: &std::path::Path,
) -> Result<Vec<LayoutBlock>, String> {
    use image::GenericImageView;
    use ort::{session::Session, value::Tensor};

    let source = image::open(image_path).map_err(|error| error.to_string())?;
    let (width, height) = source.dimensions();
    if width == 0 || height == 0 {
        return Err("image has no pixels".to_owned());
    }
    let resized = source
        .resize_exact(800, 800, image::imageops::FilterType::CatmullRom)
        .to_rgb8();
    let plane = 800 * 800;
    let mut nchw = vec![0.0_f32; plane * 3];
    for (index, pixel) in resized.pixels().enumerate() {
        nchw[index] = f32::from(pixel[0]) / 255.0;
        nchw[plane + index] = f32::from(pixel[1]) / 255.0;
        nchw[plane * 2 + index] = f32::from(pixel[2]) / 255.0;
    }

    let mut session = Session::builder()
        .map_err(|error| error.to_string())?
        .with_intra_threads(std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
        .map_err(|error| error.to_string())?
        .commit_from_file(model_path)
        .map_err(|error| error.to_string())?;
    let im_shape = Tensor::from_array(([1_usize, 2], vec![800.0_f32, 800.0]))
        .map_err(|error| error.to_string())?;
    let image =
        Tensor::from_array(([1_usize, 3, 800, 800], nchw)).map_err(|error| error.to_string())?;
    let scale = Tensor::from_array((
        [1_usize, 2],
        vec![800.0 / height as f32, 800.0 / width as f32],
    ))
    .map_err(|error| error.to_string())?;
    let outputs = session
        .run(ort::inputs![
            "im_shape" => im_shape,
            "image" => image,
            "scale_factor" => scale,
        ])
        .map_err(|error| error.to_string())?;
    let (_, rows) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|error| error.to_string())?;
    if rows.len() % 7 != 0 {
        return Err(format!("unexpected layout output length: {}", rows.len()));
    }
    let detections = rows
        .chunks_exact(7)
        .map(|row| Detection {
            class_id: row[0].max(0.0) as usize,
            score: row[1],
            left: row[2].round().clamp(0.0, width as f32),
            top: row[3].round().clamp(0.0, height as f32),
            right: row[4].round().clamp(0.0, width as f32),
            bottom: row[5].round().clamp(0.0, height as f32),
            model_order: row[6] as i32,
        })
        .collect();
    Ok(postprocess(detections))
}

#[cfg(test)]
mod tests {
    use super::{Detection, postprocess};

    #[test]
    fn postprocess_suppresses_lower_scored_overlap_and_restores_reading_order() {
        let blocks = postprocess(vec![
            Detection {
                class_id: 22,
                score: 0.7,
                left: 0.0,
                top: 0.0,
                right: 100.0,
                bottom: 100.0,
                model_order: 2,
            },
            Detection {
                class_id: 22,
                score: 0.9,
                left: 1.0,
                top: 1.0,
                right: 101.0,
                bottom: 101.0,
                model_order: 1,
            },
            Detection {
                class_id: 6,
                score: 0.8,
                left: 0.0,
                top: 120.0,
                right: 100.0,
                bottom: 150.0,
                model_order: 0,
            },
        ]);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].label, "doc_title");
        assert_eq!(blocks[0].order, 1);
        assert_eq!(blocks[1].score, 0.9);
        assert_eq!(blocks[1].order, 2);
    }

    #[test]
    #[ignore = "loads the 130 MB layout model"]
    fn real_layout_model_returns_valid_blocks() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let blocks = super::analyze(
            &root
                .join("downloads")
                .join(crate::core::MODELS[2].file_name),
            &root.join("docs/text.png"),
        )
        .unwrap();
        assert!(!blocks.is_empty());
        assert!(
            blocks
                .iter()
                .all(|block| block.right > block.left && block.bottom > block.top)
        );
    }
}
