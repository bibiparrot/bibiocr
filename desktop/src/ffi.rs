use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{DynamicImage, ImageFormat, Rgba, RgbaImage, imageops::FilterType};
use serde_json::json;
use std::{
    io::Cursor,
    net::TcpListener,
    path::Path,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[cxx::bridge(namespace = "bibiocr")]
pub mod bridge {
    struct PipelineResponse {
        markdown: String,
        layout_path: String,
        output_dir: String,
    }

    struct ImageTensorResponse {
        width: u32,
        height: u32,
        nchw: Vec<f32>,
    }

    struct ArtifactBlock {
        class_id: i32,
        label: String,
        left: f32,
        top: f32,
        right: f32,
        bottom: f32,
    }

    unsafe extern "C++" {
        include!("bibiocr/bridge.hpp");

        fn run_pipeline(
            image_path: &CxxString,
            output_dir: &CxxString,
            config_path: &CxxString,
        ) -> Result<PipelineResponse>;
    }

    extern "Rust" {
        type LlamaSession;

        fn load_image_tensor(path: &str) -> Result<ImageTensorResponse>;
        fn encode_png_crop(
            path: &str,
            left: i32,
            top: i32,
            right: i32,
            bottom: i32,
        ) -> Result<Vec<u8>>;
        fn save_image_artifacts(
            input_path: &str,
            output_dir: &str,
            stem: &str,
            blocks: &[ArtifactBlock],
        ) -> Result<()>;
        fn start_llama_server(
            executable: &str,
            model: &str,
            mmproj: &str,
            timeout_seconds: i32,
        ) -> Result<Box<LlamaSession>>;
        fn endpoint(self: &LlamaSession) -> &str;
        fn recognize(self: &LlamaSession, png: &[u8], prompt: &str) -> Result<String>;
    }
}

pub struct LlamaSession {
    child: Child,
    endpoint: String,
}

impl Drop for LlamaSession {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl LlamaSession {
    fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn recognize(&self, png: &[u8], prompt: &str) -> Result<String, String> {
        if png.is_empty() {
            return Err("cannot recognize an empty image crop".to_owned());
        }
        let body = json!({
            "model": "PaddleOCR-VL-1.6",
            "temperature": 0,
            "max_tokens": 4096,
            "stream": false,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", STANDARD.encode(png))}},
                    {"type": "text", "text": prompt}
                ]
            }]
        })
        .to_string();
        let mut response = ureq::post(format!("{}/chat/completions", self.endpoint))
            .header("Content-Type", "application/json")
            .send(&body)
            .map_err(|error| format!("llama.cpp HTTP request failed: {error}"))?;
        let value: serde_json::Value = serde_json::from_str(
            &response
                .body_mut()
                .read_to_string()
                .map_err(|error| format!("cannot read llama.cpp response: {error}"))?,
        )
        .map_err(|error| format!("llama.cpp returned invalid JSON: {error}"))?;
        value["choices"][0]["message"]["content"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| "llama.cpp response has no message content".to_owned())
    }
}

fn load_image_tensor(path: &str) -> Result<bridge::ImageTensorResponse, String> {
    let image = image::open(path).map_err(|error| format!("cannot decode input image: {error}"))?;
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return Err("input image has invalid dimensions".to_owned());
    }
    let resized = image
        .resize_exact(800, 800, FilterType::CatmullRom)
        .to_rgb8();
    let plane = 800 * 800;
    let mut nchw = vec![0.0_f32; plane * 3];
    for (index, pixel) in resized.pixels().enumerate() {
        nchw[index] = f32::from(pixel[0]) / 255.0;
        nchw[plane + index] = f32::from(pixel[1]) / 255.0;
        nchw[plane * 2 + index] = f32::from(pixel[2]) / 255.0;
    }
    Ok(bridge::ImageTensorResponse {
        width,
        height,
        nchw,
    })
}

fn encode_png_crop(
    path: &str,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
) -> Result<Vec<u8>, String> {
    let image = image::open(path).map_err(|error| format!("cannot decode input image: {error}"))?;
    let (left, top, right, bottom) = clamped_rect(&image, left, top, right, bottom)?;
    let crop = image.crop_imm(left, top, right - left, bottom - top);
    let mut bytes = Cursor::new(Vec::new());
    crop.write_to(&mut bytes, ImageFormat::Png)
        .map_err(|error| format!("cannot encode image crop: {error}"))?;
    Ok(bytes.into_inner())
}

fn save_image_artifacts(
    input_path: &str,
    output_dir: &str,
    stem: &str,
    blocks: &[bridge::ArtifactBlock],
) -> Result<(), String> {
    let source = image::open(input_path)
        .map_err(|error| format!("cannot load input image for result artifacts: {error}"))?;
    let mut overlay = source.to_rgba8();
    for block in blocks {
        let color = Rgba([
            (40 + (block.class_id * 71).rem_euclid(190)) as u8,
            (40 + (block.class_id * 47).rem_euclid(190)) as u8,
            (40 + (block.class_id * 29).rem_euclid(190)) as u8,
            255,
        ]);
        draw_rectangle(
            &mut overlay,
            block.left.round() as i32,
            block.top.round() as i32,
            block.right.round() as i32,
            block.bottom.round() as i32,
            color,
        );
    }
    let output = Path::new(output_dir);
    overlay
        .save_with_format(
            output.join(format!("{stem}_layout_det_res.png")),
            ImageFormat::Png,
        )
        .map_err(|error| format!("cannot save layout visualization: {error}"))?;
    let images = output.join("imgs");
    std::fs::create_dir_all(&images).map_err(|error| error.to_string())?;
    for block in blocks
        .iter()
        .filter(|block| matches!(block.label.as_str(), "image" | "figure" | "seal"))
    {
        let rect = clamped_rect(
            &source,
            block.left.round() as i32,
            block.top.round() as i32,
            block.right.round() as i32,
            block.bottom.round() as i32,
        )?;
        source
            .crop_imm(rect.0, rect.1, rect.2 - rect.0, rect.3 - rect.1)
            .to_rgb8()
            .save_with_format(images.join(image_filename(block)), ImageFormat::Jpeg)
            .map_err(|error| format!("cannot save extracted image block: {error}"))?;
    }
    Ok(())
}

fn start_llama_server(
    executable: &str,
    model: &str,
    mmproj: &str,
    timeout_seconds: i32,
) -> Result<Box<LlamaSession>, String> {
    for (name, path) in [
        ("llama-server", executable),
        ("model", model),
        ("mmproj", mmproj),
    ] {
        if !Path::new(path).is_file() {
            return Err(format!("{name} does not exist: {path}"));
        }
    }
    if timeout_seconds <= 0 {
        return Err("llama.cpp startup timeout must be positive".to_owned());
    }
    let listener = TcpListener::bind("127.0.0.1:0")
        .map_err(|error| format!("cannot choose a llama.cpp port: {error}"))?;
    let port = listener
        .local_addr()
        .map_err(|error| error.to_string())?
        .port();
    drop(listener);
    let port_string = port.to_string();
    let mut command = Command::new(executable);
    command
        .args([
            "-m",
            model,
            "--mmproj",
            mmproj,
            "--host",
            "127.0.0.1",
            "--port",
            &port_string,
            "--temp",
            "0",
            "--ctx-size",
            "8192",
            "--no-warmup",
            "--log-disable",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(directory) = Path::new(executable).parent() {
        command.current_dir(directory);
    }
    hide_child_window(&mut command);
    let mut child = command
        .spawn()
        .map_err(|error| format!("cannot launch llama-server: {error}"))?;
    let endpoint = format!("http://127.0.0.1:{port}/v1");
    let deadline = Instant::now() + Duration::from_secs(timeout_seconds as u64);
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!("llama-server exited during startup ({status})"));
        }
        if ureq::get(format!("http://127.0.0.1:{port}/health"))
            .call()
            .is_ok()
        {
            return Ok(Box::new(LlamaSession { child, endpoint }));
        }
        thread::sleep(Duration::from_millis(250));
    }
    let _ = child.kill();
    Err("timed out waiting for llama-server to load the models".to_owned())
}

fn clamped_rect(
    image: &DynamicImage,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
) -> Result<(u32, u32, u32, u32), String> {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let rect = (
        left.clamp(0, width) as u32,
        top.clamp(0, height) as u32,
        right.clamp(0, width) as u32,
        bottom.clamp(0, height) as u32,
    );
    if rect.2 <= rect.0 || rect.3 <= rect.1 {
        Err("layout block has invalid dimensions".to_owned())
    } else {
        Ok(rect)
    }
}

fn draw_rectangle(
    image: &mut RgbaImage,
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
    color: Rgba<u8>,
) {
    let width = image.width() as i32;
    let height = image.height() as i32;
    if width == 0 || height == 0 {
        return;
    }
    let left = left.clamp(0, width - 1);
    let right = right.clamp(0, width - 1);
    let top = top.clamp(0, height - 1);
    let bottom = bottom.clamp(0, height - 1);
    for thickness in 0..2 {
        for x in left..=right {
            for y in [top + thickness, bottom - thickness] {
                if (0..height).contains(&y) {
                    image.put_pixel(x as u32, y as u32, color);
                }
            }
        }
        for y in top..=bottom {
            for x in [left + thickness, right - thickness] {
                if (0..width).contains(&x) {
                    image.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
}

fn image_filename(block: &bridge::ArtifactBlock) -> String {
    format!(
        "img_in_{}_box_{}_{}_{}_{}.jpg",
        block.label,
        block.left.round() as i32,
        block.top.round() as i32,
        block.right.round() as i32,
        block.bottom.round() as i32
    )
}

#[cfg(windows)]
fn hide_child_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    command.creation_flags(0x0800_0000);
}

#[cfg(not(windows))]
fn hide_child_window(_command: &mut Command) {}

#[cfg(test)]
mod tests {
    use super::{bridge::ArtifactBlock, image_filename};

    #[test]
    fn artifact_filename_matches_markdown_backend() {
        let block = ArtifactBlock {
            class_id: 14,
            label: "image".to_owned(),
            left: 1.2,
            top: 2.6,
            right: 30.4,
            bottom: 40.5,
        };
        assert_eq!(image_filename(&block), "img_in_image_box_1_3_30_41.jpg");
    }
}
