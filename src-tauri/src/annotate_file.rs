use anyhow::Result;
use ndarray::Array4;
use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size},
    imgproc, prelude::*, videoio,
};
use ort::{session::Session, session::builder::GraphOptimizationLevel};
use ort::value::Tensor;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::path::Path;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use tauri::{Manager};

// Global processing state
lazy_static::lazy_static! {
    pub static ref PROCESSING_STATE: Arc<Mutex<ProcessingState>> = Arc::new(Mutex::new(ProcessingState::default()));
}

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug)]
pub struct ProcessingState {
    pub progress: f64,
    pub current_frame: i32,
    pub total_frames: i32,
    pub status: String,
    pub shots_attempted: i32,
    pub baskets_made: i32,
    pub accuracy: f64,
    pub processing_time: f64,
    pub output_path: String,
    pub error: Option<String>,
}

impl Default for ProcessingState {
    fn default() -> Self {
        Self {
            progress: 0.0,
            current_frame: 0,
            total_frames: 0,
            status: "idle".to_string(),
            shots_attempted: 0,
            baskets_made: 0,
            accuracy: 0.0,
            processing_time: 0.0,
            output_path: String::new(),
            error: None,
        }
    }
}

// ==================== CONFIGURATION ====================
struct Config;

impl Config {
    // Physics & Rules (Time in seconds)
    const SHOT_COOLDOWN: f64 = 1.5;
    const BASKET_COOLDOWN: f64 = 2.0;
    const ANIMATION_DURATION: f64 = 2.0;

    fn thresholds() -> HashMap<i32, f32> {
        let mut map = HashMap::new();
        map.insert(0, 0.6); // Ball
        map.insert(1, 0.25); // Ball in Basket
        map.insert(2, 0.7); // Player
        map.insert(3, 0.7); // Basket
        map.insert(4, 0.77); // Player Shooting
        map
    }

    fn colors() -> HashMap<i32, Scalar> {
        let mut map = HashMap::new();
        map.insert(0, Scalar::new(0.0, 165.0, 255.0, 0.0)); // Ball (Orange)
        map.insert(1, Scalar::new(0.0, 215.0, 255.0, 0.0)); // Ball in Basket (Gold)
        map.insert(2, Scalar::new(0.0, 255.0, 0.0, 0.0)); // Player (Green)
        map.insert(3, Scalar::new(0.0, 0.0, 255.0, 0.0)); // Basket (Red)
        map.insert(4, Scalar::new(255.0, 100.0, 0.0, 0.0)); // Player Shooting (Blue)
        map
    }

    fn classes() -> HashMap<i32, &'static str> {
        let mut map = HashMap::new();
        map.insert(0, "Ball");
        map.insert(1, "Ball in Basket");
        map.insert(2, "Player");
        map.insert(3, "Basket");
        map.insert(4, "Player Shooting");
        map
    }
}

// ==================== GAME STATS ====================
struct GameStats {
    // fps: f64,
    shots_attempted: i32,
    baskets_made: i32,
    shot_cooldown_frames: i32,
    basket_cooldown_frames: i32,
    anim_duration_frames: i32,
    last_shot_frame: i32,
    last_basket_frame: i32,
    basket_position: Option<(i32, i32)>,
    last_known_basket_pos: Option<(i32, i32)>,
    animation_frames: VecDeque<i32>,
}

impl GameStats {
    fn new(fps: f64) -> Self {
        let shot_cooldown_frames = (fps * Config::SHOT_COOLDOWN) as i32;
        let basket_cooldown_frames = (fps * Config::BASKET_COOLDOWN) as i32;
        let anim_duration_frames = (fps * Config::ANIMATION_DURATION) as i32;

        Self {
            // fps,
            shots_attempted: 0,
            baskets_made: 0,
            shot_cooldown_frames,
            basket_cooldown_frames,
            anim_duration_frames,
            last_shot_frame: -shot_cooldown_frames,
            last_basket_frame: -basket_cooldown_frames,
            basket_position: None,
            last_known_basket_pos: None,
            animation_frames: VecDeque::with_capacity(anim_duration_frames as usize),
        }
    }

    fn register_shot(&mut self, frame_idx: i32) -> bool {
        if frame_idx - self.last_shot_frame >= self.shot_cooldown_frames {
            self.shots_attempted += 1;
            self.last_shot_frame = frame_idx;
            return true;
        }
        false
    }

    fn register_basket(&mut self, frame_idx: i32, position: Option<(i32, i32)>) -> bool {
        if frame_idx - self.last_basket_frame >= self.basket_cooldown_frames {
            if (frame_idx - self.last_shot_frame) > (self.shot_cooldown_frames * 2) {
                self.shots_attempted += 1;
                self.last_shot_frame = frame_idx;
            }

            self.baskets_made += 1;
            self.last_basket_frame = frame_idx;
            self.basket_position = position;

            self.animation_frames.clear();
            for i in 0..self.anim_duration_frames {
                self.animation_frames.push_back(frame_idx + i);
            }
            return true;
        }
        false
    }

    fn accuracy(&self) -> f64 {
        if self.shots_attempted == 0 {
            0.0
        } else {
            (self.baskets_made as f64 / self.shots_attempted as f64) * 100.0
        }
    }

    fn get_animation_progress(&self, current_frame: i32) -> f64 {
        if !self.animation_frames.contains(&current_frame) {
            return 0.0;
        }
        let delta = current_frame - self.last_basket_frame;
        1.0 - (delta as f64 / self.anim_duration_frames as f64)
    }
}

// ==================== MODEL FUNCTIONS ====================
const INPUT_SIZE: i32 = 640;

fn load_model(model_path: &str) -> Result<Session> {
    println!("🔄 Loading ONNX Model...");

    if !Path::new(model_path).exists() {
        anyhow::bail!("❌ Model not found at {}", model_path);
    }

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    println!("✅ ONNX Model loaded successfully!");
    println!("   Input name: {}", session.inputs()[0].name());
    println!("   Output name: {}", session.outputs()[0].name());
    Ok(session)
}

fn preprocess_frame(frame: &Mat, input_size: i32) -> Result<(Array4<f32>, f32, i32, i32)> {
    let h = frame.rows();
    let w = frame.cols();
    let scale = (input_size as f32 / h as f32).min(input_size as f32 / w as f32);
    let new_h = (h as f32 * scale) as i32;
    let new_w = (w as f32 * scale) as i32;

    // Resize
    let mut resized = Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        Size::new(new_w, new_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // Create padded image
    let mut padded = Mat::new_rows_cols_with_default(
        input_size,
        input_size,
        opencv::core::CV_8UC3,
        Scalar::new(114.0, 114.0, 114.0, 0.0),
    )?;

    let pad_top = (input_size - new_h) / 2;
    let pad_left = (input_size - new_w) / 2;

    // Copy resized image into padded image at ROI
    let roi_rect = Rect::new(pad_left, pad_top, new_w, new_h);
    {
        let mut roi = padded.roi_mut(roi_rect)?;
        resized.copy_to(&mut roi)?;
    }

    // Convert BGR to RGB
    let mut rgb = Mat::default();
    #[cfg(target_os = "macos")]
    imgproc::cvt_color(&padded, &mut rgb, imgproc::COLOR_BGR2RGB, 0, core::ALGO_HINT_DEFAULT)?;
    #[cfg(not(target_os = "macos"))]
    imgproc::cvt_color(&padded, &mut rgb, imgproc::COLOR_BGR2RGB, 0)?;

    // Convert to ndarray and normalize (CHW, float32, [0,1])
    let (h, w) = (input_size as usize, input_size as usize);
    let mut img_array = Array4::<f32>::zeros((1, 3, h, w));
    let rgb_bytes = rgb.data_bytes()?;
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            // RGB order
            img_array[[0, 0, y, x]] = rgb_bytes[idx] as f32 / 255.0;
            img_array[[0, 1, y, x]] = rgb_bytes[idx + 1] as f32 / 255.0;
            img_array[[0, 2, y, x]] = rgb_bytes[idx + 2] as f32 / 255.0;
        }
    }
    Ok((img_array, scale, pad_left, pad_top))
}

fn iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x_a = box1[0].max(box2[0]);
    let y_a = box1[1].max(box2[1]);
    let x_b = box1[2].min(box2[2]);
    let y_b = box1[3].min(box2[3]);

    let inter_area = (x_b - x_a).max(0.0) * (y_b - y_a).max(0.0);
    let box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);

    inter_area / (box1_area + box2_area - inter_area)
}

fn non_max_suppression(
    boxes: Vec<[f32; 4]>,
    scores: Vec<f32>,
    classes: Vec<i32>,
    iou_threshold: f32,
    score_threshold: f32,
) -> (Vec<[f32; 4]>, Vec<f32>, Vec<i32>) {
    if boxes.is_empty() {
        return (vec![], vec![], vec![]);
    }

    // Filter by score threshold
    let filtered: Vec<_> = boxes
        .into_iter()
        .zip(scores)
        .zip(classes)
        .filter(|((_, score), _)| *score >= score_threshold)
        .map(|((box_, score), class)| (box_, score, class))
        .collect();

    if filtered.is_empty() {
        return (vec![], vec![], vec![]);
    }

    let mut keep_boxes = Vec::new();
    let mut keep_scores = Vec::new();
    let mut keep_classes = Vec::new();

    // Get unique classes
    let mut unique_classes: Vec<i32> = filtered.iter().map(|(_, _, c)| *c).collect();
    unique_classes.sort_unstable();
    unique_classes.dedup();

    for cls in unique_classes {
        let mut cls_items: Vec<_> = filtered
            .iter()
            .filter(|(_, _, c)| *c == cls)
            .cloned()
            .collect();

        // Sort by score descending
        cls_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut active = vec![true; cls_items.len()];

        for i in 0..cls_items.len() {
            if !active[i] {
                continue;
            }

            keep_boxes.push(cls_items[i].0);
            keep_scores.push(cls_items[i].1);
            keep_classes.push(cls_items[i].2);

            let current_box = &cls_items[i].0;

            for j in (i + 1)..cls_items.len() {
                if !active[j] {
                    continue;
                }

                let iou_value = iou(current_box, &cls_items[j].0);
                if iou_value > iou_threshold {
                    active[j] = false;
                }
            }
        }
    }

    (keep_boxes, keep_scores, keep_classes)
}

fn postprocess_output(
    output: ndarray::ArrayView2<'_, f32>,
    orig_shape: (i32, i32),
    scale: f32,
    pad_left: i32,
    pad_top: i32,
) -> (Vec<[f32; 4]>, Vec<f32>, Vec<i32>) {
    use ndarray::Axis;
    let num_detections = output.shape()[0];
    let num_attrs = output.shape()[1];
    let mut boxes = Vec::new();
    let mut scores = Vec::new();
    let mut classes = Vec::new();
    for i in 0..num_detections {
        let row = output.index_axis(Axis(0), i);
        let x_center = row[0];
        let y_center = row[1];
        let width = row[2];
        let height = row[3];
        // Support both 8 and 9 attribute outputs (4 box + N class scores)
        let class_scores = &row.as_slice().unwrap()[4..num_attrs];
        let (max_class, max_score) = class_scores
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |(max_idx, max_val), (idx, &val)| {
                if val > max_val {
                    (idx, val)
                } else {
                    (max_idx, max_val)
                }
            });
        let mut x1 = x_center - width / 2.0;
        let mut y1 = y_center - height / 2.0;
        let mut x2 = x_center + width / 2.0;
        let mut y2 = y_center + height / 2.0;
        x1 = (x1 - pad_left as f32) / scale;
        y1 = (y1 - pad_top as f32) / scale;
        x2 = (x2 - pad_left as f32) / scale;
        y2 = (y2 - pad_top as f32) / scale;
        let (h, w) = orig_shape;
        x1 = x1.max(0.0).min(w as f32);
        y1 = y1.max(0.0).min(h as f32);
        x2 = x2.max(0.0).min(w as f32);
        y2 = y2.max(0.0).min(h as f32);
        boxes.push([x1, y1, x2, y2]);
        scores.push(max_score);
        classes.push(max_class as i32);
    }
    non_max_suppression(boxes, scores, classes, 0.45, 0.25)
}

// ==================== VISUALIZATION ====================
fn draw_yolo_boxes(
    frame: &mut Mat,
    boxes: &[[f32; 4]],
    scores: &[f32],
    classes: &[i32],
) -> Result<()> {
    if boxes.is_empty() {
        return Ok(());
    }

    let thresholds = Config::thresholds();
    let colors = Config::colors();
    let class_names = Config::classes();

    for i in 0..boxes.len() {
        let cls = classes[i];
        let conf = scores[i];

        if conf < *thresholds.get(&cls).unwrap_or(&0.3) {
            continue;
        }

        let box_ = &boxes[i];
        let (x1, y1, x2, y2) = (box_[0] as i32, box_[1] as i32, box_[2] as i32, box_[3] as i32);
        let color_val = colors.get(&cls).unwrap_or(&Scalar::new(255.0, 255.0, 255.0, 0.0)).clone();
        let label = format!("{} {:.2}", class_names.get(&cls).unwrap_or(&"Unknown"), conf);

        let rect = Rect::new(x1, y1, (x2 - x1).max(1), (y2 - y1).max(1));
        imgproc::rectangle(
            frame,
            rect,
            color_val,
            2,
            imgproc::LINE_8,
            0,
        )?;

        imgproc::put_text(
            frame,
            &label,
            Point::new(x1, y1 - 10),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            color_val,
            2,
            imgproc::LINE_8,
            false,
        )?;
    }

    Ok(())
}

fn draw_basket_effect(frame: &mut Mat, position: Option<(i32, i32)>, progress: f64) -> Result<()> {
    if let Some((cx, cy)) = position {
        if progress > 0.0 {
            let max_radius = 80;
            let radius = (max_radius as f64 * progress) as i32;
            let alpha = (255.0 * (1.0 - progress)) as i32;

            let color = Scalar::new(0.0, 215.0, 255.0, alpha as f64);
            imgproc::circle(
                frame,
                Point::new(cx, cy),
                radius,
                color,
                3,
                imgproc::LINE_8,
                0,
            )?;
        }
    }
    Ok(())
}

fn draw_hud(frame: &mut Mat, stats: &GameStats, _w: i32, _h: i32) -> Result<()> {
    // Draw semi-transparent panel
    let mut overlay = frame.clone();
    let rect = Rect::new(10, 10, 240, 100);
    imgproc::rectangle(
        &mut overlay,
        rect,
        Scalar::new(0.0, 0.0, 0.0, 180.0),
        -1,
        imgproc::LINE_8,
        0,
    )?;

    // Blend
    // To avoid mutable/immutable borrow, use a temp Mat
    let mut blended = frame.clone();
    core::add_weighted(&overlay, 0.7, frame, 0.3, 0.0, &mut blended, -1)?;
    blended.copy_to(frame)?;

    // Draw text
    let white = Scalar::new(255.0, 255.0, 255.0, 0.0);
    imgproc::put_text(
        frame,
        &format!("Shots: {}", stats.shots_attempted),
        Point::new(20, 35),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        white,
        2,
        imgproc::LINE_8,
        false,
    )?;

    imgproc::put_text(
        frame,
        &format!("Baskets: {}", stats.baskets_made),
        Point::new(20, 65),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        white,
        2,
        imgproc::LINE_8,
        false,
    )?;

    imgproc::put_text(
        frame,
        &format!("Accuracy: {:.1}%", stats.accuracy()),
        Point::new(20, 95),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        white,
        2,
        imgproc::LINE_8,
        false,
    )?;

    Ok(())
}

fn draw_final_screen(
    w: i32,
    h: i32,
    stats: &GameStats,
    total_frames: i32,
    fps: f64,
    processing_time: f64,
) -> Result<Mat> {
    let mut frame = Mat::new_rows_cols_with_default(
        h,
        w,
        opencv::core::CV_8UC3,
        Scalar::new(26.0, 26.0, 26.0, 0.0),
    )?;

    let white = Scalar::new(255.0, 255.0, 255.0, 0.0);

    imgproc::put_text(
        &mut frame,
        "GAME STATS",
        Point::new(w / 2 - 200, h / 4),
        imgproc::FONT_HERSHEY_SIMPLEX,
        2.0,
        white,
        3,
        imgproc::LINE_8,
        false,
    )?;

    let stats_y = h / 2 - 80;
    imgproc::put_text(
        &mut frame,
        &format!("Shots Attempted: {}", stats.shots_attempted),
        Point::new(w / 2 - 300, stats_y),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.2,
        white,
        2,
        imgproc::LINE_8,
        false,
    )?;

    imgproc::put_text(
        &mut frame,
        &format!("Baskets Made: {}", stats.baskets_made),
        Point::new(w / 2 - 300, stats_y + 50),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.2,
        white,
        2,
        imgproc::LINE_8,
        false,
    )?;

    imgproc::put_text(
        &mut frame,
        &format!("Accuracy: {:.1}%", stats.accuracy()),
        Point::new(w / 2 - 300, stats_y + 100),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.2,
        white,
        2,
        imgproc::LINE_8,
        false,
    )?;

    imgproc::put_text(
        &mut frame,
        &format!(
            "Processing Time: {:.2}s | Frames: {} | FPS: {:.1}",
            processing_time, total_frames, fps
        ),
        Point::new(50, h - 50),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        white,
        1,
        imgproc::LINE_8,
        false,
    )?;

    Ok(frame)
}

// ==================== VIDEO PROCESSING ====================
fn process_detections(
    boxes: &[[f32; 4]],
    scores: &[f32],
    classes: &[i32],
    stats: &mut GameStats,
    frame_idx: i32,
) {
    if boxes.is_empty() {
        return;
    }

    let thresholds = Config::thresholds();

    for i in 0..boxes.len() {
        let cls = classes[i];
        let conf = scores[i];

        if conf < *thresholds.get(&cls).unwrap_or(&0.3) {
            continue;
        }

        let box_ = &boxes[i];
        let center = (
            ((box_[0] + box_[2]) / 2.0) as i32,
            ((box_[1] + box_[3]) / 2.0) as i32,
        );

        match cls {
            3 => {
                // Basket
                stats.last_known_basket_pos = Some(center);
            }
            4 => {
                // Player Shooting
                stats.register_shot(frame_idx);
            }
            1 => {
                // Ball in Basket
                let target_pos = stats.last_known_basket_pos.or(Some(center));
                stats.register_basket(frame_idx, target_pos);
            }
            _ => {}
        }
    }
}

fn process_video(app: tauri::AppHandle, input_path: &str, output_path: &str) -> Result<()> {
    println!("\n🎬 Starting video processing...");
    println!("📁 Current working directory: {:?}", env::current_dir()?);
    println!("📁 Input:  {}", input_path);
    println!("📁 Output: {}", output_path);

    // Initialize processing state
    {
        let mut state = PROCESSING_STATE.lock().unwrap();
        *state = ProcessingState {
            progress: 0.0,
            current_frame: 0,
            total_frames: 0,
            status: "Initializing...".to_string(),
            shots_attempted: 0,
            baskets_made: 0,
            accuracy: 0.0,
            processing_time: 0.0,
            output_path: output_path.to_string(),
            error: None,
        };
    }

    let start_time = Instant::now();

    // Load model - use Tauri's resource path resolver to find bundled model
    let model_path = app
        .path()
        .resolve("_up_/models/yolo11n-detect.onnx", tauri::path::BaseDirectory::Resource)
        .map_err(|e| anyhow::anyhow!("Failed to resolve model path: {}", e))?;

    let model_path_str = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid model path"))?;

    println!("📦 Model path: {}", model_path_str);
    let mut session = load_model(model_path_str)?;

    // Open video
    let mut cap = videoio::VideoCapture::from_file(input_path, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cap)? {
        anyhow::bail!("❌ Could not open video file.");
    }

    // Get video properties
    let fps = cap.get(videoio::CAP_PROP_FPS)?;
    let w = cap.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
    let h = cap.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
    let total_frames = cap.get(videoio::CAP_PROP_FRAME_COUNT)? as i32;

    println!("📊 Video: {}x{} @ {:.2} FPS, {} frames", w, h, fps, total_frames);

    // Create video writer
    let fourcc = videoio::VideoWriter::fourcc('m', 'p', '4', 'v')?;
    let mut writer = videoio::VideoWriter::new(
        output_path,
        fourcc,
        fps,
        Size::new(w, h),
        true,
    )?;

    // Initialize game stats
    let mut stats = GameStats::new(fps);
    let mut frame_idx = 0;

    println!("⚙️  Processing frames...");


    // Get input name before frame loop to avoid borrow checker issues
    let input_name = session.inputs()[0].name().to_string();

    // Process each frame
    let mut frame = Mat::default();
    while frame_idx < total_frames {
        let success = cap.read(&mut frame)?;
        if !success || frame.empty() {
            break;
        }

        let mut annotated = frame.clone();

        // Preprocess frame
        let (input_tensor, scale, pad_left, pad_top) = preprocess_frame(&frame, INPUT_SIZE)?;

        // Prepare ONNX input
        let input_vec: Vec<f32> = input_tensor.iter().cloned().collect();
        let input_shape = input_tensor.shape();
        let input_tensor_ort = Tensor::from_array((input_shape, input_vec))?;
        let mut input_map = std::collections::HashMap::new();
        input_map.insert(input_name.as_str(), input_tensor_ort);
        let outputs = session.run(input_map)?;
        let (output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()?;

        // Robust output shape handling (align with Python/Node.js)
        // Output can be [1, 8/9, 8400] or [1, 8400, 8/9]
        let (num0, num1, num2) = match output_shape.len() {
            3 => (output_shape[0], output_shape[1], output_shape[2]),
            2 => (output_shape[0], output_shape[1], 1),
            _ => return Err(anyhow::anyhow!("Unexpected output shape: {:?}", output_shape)),
        };
        let (flat, n, m) = if num0 == 1 && (num1 == 8 || num1 == 9) {
            // [1, 8/9, 8400] -> transpose to [8400, 8/9]
            let n = num2 as usize;
            let m = num1 as usize;
            let mut arr = vec![0f32; n * m];
            for i in 0..n {
                for j in 0..m {
                    arr[i * m + j] = output_data[(j * n + i) as usize];
                }
            }
            (arr, n, m)
        } else if num0 == 1 && (num2 == 8 || num2 == 9) {
            // [1, 8400, 8/9] -> already correct
            let n = num1 as usize;
            let m = num2 as usize;
            (output_data.to_vec(), n, m)
        } else {
            return Err(anyhow::anyhow!("Unsupported output shape: {:?}", output_shape));
        };
        let output = ndarray::ArrayView2::from_shape((n, m), &flat).unwrap();

        // Postprocess outputs
        let (boxes, scores, classes) = postprocess_output(output, (h, w), scale, pad_left, pad_top);

        // Update statistics
        process_detections(&boxes, &scores, &classes, &mut stats, frame_idx);

        // Draw bounding boxes
        draw_yolo_boxes(&mut annotated, &boxes, &scores, &classes)?;

        // Draw basket effect animation
        let anim_progress = stats.get_animation_progress(frame_idx);
        if anim_progress > 0.0 {
            draw_basket_effect(&mut annotated, stats.basket_position, anim_progress)?;
        }

        // Draw HUD
        draw_hud(&mut annotated, &stats, w, h)?;

        // Write frame
        writer.write(&annotated)?;
        frame_idx += 1;

        // Progress update every second (based on FPS)
        if frame_idx % fps.round() as i32 == 0 || frame_idx % 30 == 0 {
            let progress = (frame_idx as f64 / total_frames as f64) * 100.0;
            let elapsed = start_time.elapsed().as_secs_f64();

            print!("   Progress: {:.1}% ({}/{})\r", progress, frame_idx, total_frames);

            // Update global state
            {
                let mut state = PROCESSING_STATE.lock().unwrap();
                state.progress = progress;
                state.current_frame = frame_idx;
                state.total_frames = total_frames;
                state.status = "Processing frames...".to_string();
                state.shots_attempted = stats.shots_attempted;
                state.baskets_made = stats.baskets_made;
                state.accuracy = stats.accuracy();
                state.processing_time = elapsed;
            }
        }
    }

    println!("\n✅ Frame processing complete!");
    println!(
        "📈 Stats: {} shots, {} baskets, {:.1}% accuracy",
        stats.shots_attempted,
        stats.baskets_made,
        stats.accuracy()
    );

    // Add final summary screen (5 seconds)
    // Calculate processing time before creating summary frame so it can be displayed
    let processing_time = start_time.elapsed().as_secs_f64();
    let summary_frame = draw_final_screen(w, h, &stats, frame_idx, fps, processing_time)?;
    for _ in 0..(fps * 5.0) as i32 {
        writer.write(&summary_frame)?;
    }

    // Cleanup
    cap.release()?;
    writer.release()?;

    // Calculate final processing time after all operations
    let total_processing_time = start_time.elapsed().as_secs_f64();

    println!("\n{}", "=".repeat(60));
    println!("🎉 PROCESSING COMPLETE!");
    println!("{}", "=".repeat(60));
    println!("⏱️  Total Processing Time: {:.2} seconds", total_processing_time);
    println!("📊 Shots Attempted: {}", stats.shots_attempted);
    println!("🏀 Baskets Made: {}", stats.baskets_made);
    println!("🎯 Accuracy: {:.1}%", stats.accuracy());
    println!("📁 Output saved to: {}", output_path);
    println!("{}\n", "=".repeat(60));

    // Update final state
    {
        let mut state = PROCESSING_STATE.lock().unwrap();
        state.progress = 100.0;
        state.current_frame = total_frames;
        state.total_frames = total_frames;
        state.status = "Complete!".to_string();
        state.shots_attempted = stats.shots_attempted;
        state.baskets_made = stats.baskets_made;
        state.accuracy = stats.accuracy();
        state.processing_time = total_processing_time;
        state.output_path = output_path.to_string();
    }

    Ok(())
}

// ==================== MAIN ====================
pub fn annotate_video(app: tauri::AppHandle, file_path: String, output_path: String) -> Result<()> {
    let input_path = &file_path;
    let output_path_str = &output_path;

    if !Path::new(input_path).exists() {
        anyhow::bail!("❌ Error: Input file '{}' not found!", input_path);
    }

    process_video(app, input_path, output_path_str)?;

    Ok(())
}
